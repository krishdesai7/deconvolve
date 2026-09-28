"""Optional wall-clock instrumentation for a run's phases.

Off unless `DECONVOLVE_TIMING` is set, and *off* means a shared no-op context manager:
no `perf_counter`, no allocation, nothing appended. That matters because the
timers sit at phase boundaries inside `workflows.train.run` and `engine.train`, which a
sweep crosses a few hundred times.

The point of the layer is to say which component to go optimize, so the report
is shares of the total rather than raw seconds alone, and nested phases are
recorded with their depth so `train.compile` can be read against `train`.

One number needs a caveat carried with it. `train.compile` reads near-zero
whenever XLA's persistent cache is warm, which
is the common case and would point optimization effort at the wrong place.
`timings.json` therefore records whether the cache directory held anything when
the run started, sampled before the first compile could fill it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast, override

from rich.console import Console
from rich.table import Table

from ..coretypes import COMPILE_CACHE_DIR, artifacts_dir

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from logging import Logger
    from pathlib import Path
    from typing import Any, LiteralString

logger: Logger = logging.getLogger(name=__name__)

TIMING_ENV_VAR: Final[LiteralString] = "DECONVOLVE_TIMING"

# Spelled out rather than `bool(value)`, because the string "0" is truthy and a
# SLURM `--export` that forwards an unset variable delivers "" rather than absence.
_FALSEY: Final[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})


def _enabled_from_env(environ: Mapping[str, str], /) -> bool:
    return environ.get(TIMING_ENV_VAR, "").strip().lower() not in _FALSEY


@dataclass(frozen=True, slots=True)
class Phase:
    """One completed phase. `depth` is 0 for a top-level phase."""

    name: str
    seconds: float
    depth: int
    detail: str | None = None
    failed: bool = False


class _Open:
    """The handle a live phase yields. One instance per open phase."""

    __slots__ = ("detail",)

    def __init__(self) -> None:
        self.detail: str | None = None

    def note(self, detail: str, /) -> None:
        """Annotate the phase, e.g. which branch of a cache check it took."""
        self.detail: str = detail

    def block[T](self, value: T, /) -> T:
        """Wait for JAX to finish producing `value`, inside the clock.

        JAX is async, so a timer stopped before the arrays are ready charges
        this phase's time to whichever phase runs next. This shifts *when* the
        wait happens and never what is computed; with timing off it does not
        happen at all, which is why it lives on the handle rather than at the
        call site.
        """
        import jax

        return cast(typ="T", val=jax.block_until_ready(x=value))


class _Noop(_Open):
    """The single instance handed out when timing is off."""

    __slots__ = ()

    @override
    def note(self, detail: str, /) -> None:
        del detail

    @override
    def block[T](self, value: T, /) -> T:
        return value


_NOOP: Final[_Noop] = _Noop()


class _Recorder:
    """Completed phases, in closing order, plus the live nesting depth."""

    __slots__ = ("compile_cache_warm", "depth", "names", "open", "records")

    def __init__(self) -> None:
        self.records: list[Phase] = []
        self.open: list[_Open] = []
        self.names: list[str] = []
        self.depth: int = 0
        # Sampled at construction, before run has had a chance to compile into it.
        self.compile_cache_warm: bool = any(COMPILE_CACHE_DIR.glob(pattern="*"))


_recorder: _Recorder | None = _Recorder() if _enabled_from_env(os.environ) else None


def is_enabled() -> bool:
    return _recorder is not None


def enable(active: bool = True, /) -> None:
    """Turn timing on or off and discard anything already recorded.

    The environment decides this at import; this exists for tests and for a
    caller that knows better than the environment does.
    """
    global _recorder
    _recorder = _Recorder() if active else None


def reset() -> None:
    """Discard recorded phases between passes, leaving enabled/disabled as is.

    `write` merges a pass's phases into whatever is already on disk, so a
    caller that runs more than one pass in the same process (a test, or a
    launcher that never re-execs between phases) needs a way to start the next
    pass's recording clean without also flipping timing off. A no-op when
    timing is off, since there is nothing to discard.
    """
    global _recorder
    if _recorder is not None:
        _recorder = _Recorder()


def phases() -> tuple[Phase, ...]:
    """Completed phases in closing order: children before their parent."""
    return () if _recorder is None else tuple(_recorder.records)


def note(detail: str, /, *, to: str | None = None) -> None:
    """Annotate an open phase, if there is one.

    Lets `datasets.py` say "cache hit" about a phase that `workflows/train.py`
    opened, without the loaders having to own a phase of their own or thread
    a handle down through their signatures.

    `to` names which open phase it means, and the loaders always pass it. They
    are called from more than one place --- `evaluate_run` rebuilds the same
    dataset inside the `evaluate` phase --- and annotating the innermost open
    phase would put "cache hit" in the detail column of a row about metrics.
    Unnamed, or named for a phase that is not open, the note is dropped.
    """
    if _recorder is None or not _recorder.open:
        return
    if to is None:
        _recorder.open[-1].note(detail)
        return
    for handle, name in zip(
        reversed(_recorder.open), reversed(_recorder.names), strict=True
    ):
        if name == to:
            handle.note(detail)
            return


@contextmanager
def phase(name: str, /, *, detail: str | None = None) -> Generator[_Open]:
    """Time the block, unless timing is off -- in which case do nothing at all."""
    recorder: _Recorder | None = _recorder
    if recorder is None:
        yield _NOOP
        return

    handle = _Open()
    handle.detail = detail
    depth: int = recorder.depth
    recorder.open.append(handle)
    recorder.names.append(name)
    recorder.depth = depth + 1
    start: float = time.perf_counter()
    failed = False
    try:
        yield handle
    except BaseException:
        # The phase that just fell over is the one whose number is most worth
        # having, so record it and re-raise rather than losing it.
        failed = True
        raise
    finally:
        elapsed: float = time.perf_counter() - start
        recorder.depth = depth
        _ = recorder.open.pop()
        _ = recorder.names.pop()
        recorder.records.append(
            Phase(
                name=name,
                seconds=elapsed,
                depth=depth,
                detail=handle.detail,
                failed=failed,
            )
        )


def record(name: str, seconds: float, /, *, detail: str | None = None) -> None:
    """Record a phase this process could not time itself.

    `phase()` covers everything this interpreter runs, which is almost
    everything. The exception is the OmniFold baseline: its real work happens
    in a subprocess under a different Python, and the breakdown comes back as
    numbers in an `.npz` rather than as a block to wrap. Those are still phases
    of the run and belong in the same table, so this is the way in.

    Depth follows the current nesting, so a call inside `with phase("omnifold")`
    lands underneath it exactly as a nested `phase()` would. Ordering works out
    for the same reason `phase()`'s does: `_ordered` reads a top-level phase's
    children off the records that landed before it closed.

    A no-op when timing is off, like everything else here.
    """
    recorder: _Recorder | None = _recorder
    if recorder is None:
        return
    recorder.records.append(
        Phase(name=name, seconds=seconds, depth=recorder.depth, detail=detail)
    )


def _total_seconds(records: list[Phase], /) -> float:
    """Sum of the top-level phases only; a nested one is already inside its parent."""
    return sum(p.seconds for p in records if p.depth == 0)


def _ordered(records: list[Phase], /) -> list[Phase]:
    """Closing order puts children before parents. Read it back parents-first."""
    ordered: list[Phase] = []
    for i, record in enumerate(records):
        if record.depth != 0:
            continue
        ordered.append(record)
        # A top-level phase's children are the records that closed before it and
        # after the previous top-level one.
        start: int = next(
            (j + 1 for j in range(i - 1, -1, -1) if records[j].depth == 0), 0
        )
        ordered.extend(records[start:i])
    return ordered


def report(console: Console | None = None, /) -> None:
    """Print the phase table. Silent when timing is off or nothing was timed."""
    if _recorder is None or not _recorder.records:
        return
    total: float = _total_seconds(_recorder.records)
    table = Table(title="Wall clock by phase")
    table.add_column(header="Phase")
    table.add_column(header="Seconds", justify="right")
    table.add_column(header="Share", justify="right")
    table.add_column(header="Detail")
    for record in _ordered(_recorder.records):
        share: str = f"{100.0 * record.seconds / total:.1f}%" if total > 0 else "-"
        detail: str = record.detail or ""
        if record.failed:
            detail = f"{detail} (raised)".strip()
        table.add_row(
            "  " * record.depth + record.name,
            f"{record.seconds:.3f}",
            share,
            detail,
        )
    (console or Console()).print(table)


def _is_valid_phase(phase: object, /) -> bool:
    """Whether a parsed phase record has a name to merge on, a depth to sum
    by, and a finite number to sum -- what `_merged_phases` and the total
    need to read it without raising.

    `NaN` and `Infinity` are `float`s and pass the isinstance check, but a
    payload carrying either is corrupt in exactly the way the other invalid
    shapes are: one of them poisons `total_seconds` for every phase in the
    file, silently and irrecoverably. Rejected here rather than tolerated.
    """
    if not isinstance(phase, dict):
        return False
    seconds: object = phase.get("seconds")
    return (
        isinstance(phase.get("name"), str)
        and isinstance(phase.get("depth"), int)
        and isinstance(seconds, int | float)
        and math.isfinite(seconds)
    )


def _is_valid_payload(payload: object, /) -> bool:
    """Whether a parsed `timings.json` has the shape `write` needs.

    Valid JSON in the wrong shape --- a bare list, a `phases` entry missing
    `depth`, a non-numeric `seconds` --- would otherwise raise inside
    `_merged_phases` or the total, from exactly the `finally` block this layer
    must never take down.
    """
    if not isinstance(payload, dict):
        return False
    phases: object = payload.get("phases", [])
    if not isinstance(phases, list):
        return False
    return all(_is_valid_phase(p) for p in cast(typ="list[object]", val=phases))


def _existing(path: Path, /) -> dict[str, Any]:
    """What is already on disk, or an empty payload.

    A malformed file --- unparseable text, or well-formed JSON in the wrong
    shape --- is treated as absent rather than raised on: this layer exists to
    describe a run, and must never be what ends one.
    """
    try:
        payload: Any = json.loads(s=path.read_text())
    except (OSError, ValueError):
        return {}
    return cast(typ="dict[str, Any]", val=payload) if _is_valid_payload(payload) else {}


def _merged_phases(
    previous: dict[str, Any], fresh: list[dict[str, Any]], /
) -> list[dict[str, Any]]:
    """`fresh` replaces any same-named record from `previous`; the rest survives."""
    replaced: frozenset[str] = frozenset(p["name"] for p in fresh)
    kept: list[dict[str, Any]] = [
        p
        for p in cast(typ="list[dict[str, Any]]", val=previous.get("phases", []))
        if p["name"] not in replaced
    ]
    return kept + fresh


def write(run_dir: Path, /, *, pass_name: str, filename: str = "timings.json") -> None:
    """Merge this pass's phases into `filename`. A no-op when timing is off
    or nothing was timed.

    `scripts/submit.zsh` makes three passes over one run directory -- train,
    baseline, then reload for the figures -- so an overwriting writer would
    let the reload pass destroy the training numbers on every pipeline run.
    Phases merge by name instead: this pass's record replaces a same-named one
    from an earlier pass and leaves the rest untouched. `pass_name` names
    which invocation produced each row.

    Flat, with a `depth` field rather than nested objects, so a sweep can join
    it against `config.json` without walking a tree. Every number here comes
    from `perf_counter`, so the `np.float32` JSON hazard cannot arise --
    nothing needs coercing on the way out.

    `filename` exists because the merge is **by phase name alone, not by
    (pass, name)**. That is right for the passes of one pipeline over one
    run: `load` legitimately replaces `train`'s `plots` row. It is wrong for
    a different program over the same directory.
    `deconvolve baseline omnifold` also has phases called `data` and `evaluate`, and
    writing them here would silently destroy the training pass's --- the rows
    anyone actually wants. So it writes `timings_omnifold.json` instead, and
    the baseline's cost stays separable from the method's, which is the
    comparison the numbers are for.
    """
    if _recorder is None or not _recorder.records:
        return
    try:
        path: Path = artifacts_dir(run_dir) / filename
    except OSError as error:
        # An unwritable run directory must not take the run down over a
        # report of its own timing -- log and move on.
        logger.warning("Could not create %s: %s", run_dir, error)
        return
    previous: dict[str, Any] = _existing(path)
    fresh: list[dict[str, Any]] = [
        {
            "name": p.name,
            "seconds": p.seconds,
            "depth": p.depth,
            "detail": p.detail,
            "failed": p.failed,
            "pass": pass_name,
        }
        for p in _ordered(_recorder.records)
    ]
    phases: list[dict[str, Any]] = _merged_phases(previous, fresh)
    # `compile_cache_warm` is only meaningful for a pass that actually
    # compiled something -- a reload pass samples it anyway (every
    # `_Recorder` does, unconditionally), but that reading says nothing
    # about whether *this* pass's compile deconvolve warm or cold, because it never
    # had a `compile` phase to begin with. Keep the training pass's reading
    # in that case rather than overwrite it with an irrelevant one.
    compiled_this_pass: bool = any(p.name == "compile" for p in _recorder.records)
    warm: bool = (
        _recorder.compile_cache_warm
        if compiled_this_pass or "compile_cache_warm" not in previous
        else previous["compile_cache_warm"]
    )
    payload: dict[str, Any] = {
        "total_seconds": sum(p["seconds"] for p in phases if p["depth"] == 0),
        "compile_cache_warm": warm,
        "phases": phases,
    }
    try:
        _ = path.write_text(data=json.dumps(obj=payload, indent=2))
    except OSError as error:
        # An unwritable or full run directory must not take the run down over
        # a report of its own timing -- log and move on.
        logger.warning("Could not write %s: %s", path, error)
