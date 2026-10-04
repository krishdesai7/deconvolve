"""Compile a run directory into one LaTeX dossier.

`config.json`, `metrics.json` and `timings.json` are machine interfaces. This module is
a read-only consumer that produces a human-readable report. All rounding policy lives in
the template, in siunitx col specs. A column formatted string-by-string in Python cannot
align decimal markers so this file emits full precision and lets LaTeX handle formatting
"""

from __future__ import annotations

import json
import logging
import re
import shutil

# One fixed argv, no shell, and the only interpolated element is a path
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from importlib import resources
from itertools import starmap
from typing import TYPE_CHECKING, cast

import numpy as np

from ..coretypes import (
    ARTIFACTS_DIR,
    JET_OBS,
    JET_VARIABLE_GROUPS,
    JOINT_METRICS_KEY,
    METRIC_DISPLAY_SCALE,
    artifacts_dir,
    display_order,
    figure_pages,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import Any, Final

logger: Logger = logging.getLogger(name=__name__)

# `<<[A-Z_]+>>` rather than bare `<<`: the template's header comment documents the
# replacement contract and contains `<<...>>`.
TEMPLATE_TOKEN: Final[re.Pattern[str]] = re.compile(pattern=r"<<[A-Z_]+>>")

_LATEX_SPECIALS: Final[dict[str, str]] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Enough places for four sig figs of the smallest metric a run produces (~8.294e-5). The
# template's D column is `table-format=2.8`. It also sets a resolution floor: anything
# below ~5e-11 rounds away entirely and renders as string "0" rather than a small number
_DECIMAL_PLACES: Final[int] = 10

# Numeric cell shows "---" when it has no number to show.
_DASH: Final[str] = r"\multicolumn{1}{c}{---}"


def load_template() -> str:
    """The shipped LaTeX template, as text."""
    return (
        resources.files(anchor="deconvolve.reporting") / "templates" / "report.tex"
    ).read_text(encoding="utf-8")


def _plain(value: float, /) -> str:
    """Fixed-point text for a finite value; no finiteness check of its own."""
    rendered: str = f"{value:.{_DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def decimal(value: float, /) -> str:
    """Plain decimal notation; `_DASH` when not finite. Every numeric cell lands in a
    siunitx `S` column. `nan` and `inf` must be handled here
    """
    if not np.isfinite(value):
        return _DASH
    return _plain(value)


def _num(macro: str, value: float, /) -> str:
    r"""`\<macro>{n}`, or a bare dash when there is no number to wrap.

    `\Raw` and `\Count` expand to siunitx `\num`, which rejects a non-finite
    argument the dash has to *replace* the wrapper rather than sit inside it.
    """
    rendered: str = decimal(value)
    return rendered if rendered == _DASH else rf"\{macro}{{{rendered}}}"


def latex_text(value: str, /) -> str:
    """Escape the characters that would end a compile or change the meaning."""
    return "".join(_LATEX_SPECIALS.get(character, character) for character in value)


# Keys whose values span the full row (`\ConfigWide`) rather than pairing two entries
# per line (`\ConfigPair`): a variable list, a Gaussian's raw parameters, and the MMD
# bandwidth brackets, all of which are too long or too structured for a narrow cell.
_WIDE_KEYS: Final[frozenset[str]] = frozenset(
    {"variables", "mmd_sigmas_detector", "mmd_sigmas_particle", "gaussian_params"}
)


def _scalar_cell(value: object, /) -> str:
    r"""One `\ConfigPair` cell for a scalar config value."""
    match value:
        case bool():
            return latex_text(str(object=value))
        case int():
            return rf"\Count{{{value}}}"
        case float():
            return _num("Raw", value)
        case _:
            return rf"\ConfigVal{{{value}}}"


def _format_value(value: object, /) -> str:
    r"""Render a (possibly nested) JSON-ish value for a `\ConfigVal` cell."""
    match value:
        case list():
            items: list[object] = list(value)
            return "[" + ", ".join(_format_value(item) for item in items) + "]"
        case float():
            # `_plain`, since this lands inside `\ConfigVal`=`\texttt{\detokenize{...}}`
            return _plain(value) if np.isfinite(value) else str(object=value)
        case _:
            return str(object=value)


def _populated_groups(present: frozenset[str], /) -> list[tuple[str, Sequence[str]]]:
    """The display groups this run's variables actually populate, in order.

    `--var m --var w` leaves the splitting group empty, and an empty group would print a
    heading with no rows under it.
    """
    return [
        (label, in_group)
        for label, members in JET_VARIABLE_GROUPS
        if (in_group := [v for v in members if v in present])
    ]


def _symbol(name: str, /) -> str:
    """A variable's LaTeX symbol: the jet observable's if known, else its raw name."""
    return JET_OBS[name].symbol if name in JET_OBS else latex_text(name)


def _group_line(label: str, members: Sequence[str], /) -> str:
    r"""One `\textbf{<label>:} <symbols>` line for a populated group."""
    symbols: str = ", ".join(_symbol(name) for name in members)
    return rf"\textbf{{{label}:}} {symbols}"


def _grouped_variables_cell(
    names: Sequence[str], groups: Sequence[tuple[str, Sequence[str]]], /
) -> str:
    r"""The multiline `\textbf{<label>:} <symbols>` cell for a jet run's groups."""
    formatted: list[str] = list(starmap(_group_line, groups))
    grouped_names: frozenset[str] = frozenset(
        v for _, members in groups for v in members
    )
    other: list[str] = [v for v in names if v not in grouped_names]
    if other:
        formatted.append(_group_line("Other observables", other))
    return r" \newline\vspace{2pt} ".join(formatted)


def _variables_cell(names: Sequence[str], /) -> str:
    r"""The observable list formatted by physics group, or flat if non-jet.

    Each populated group is rendered multiline: `\textbf{<label>:} <symbols>`.
    """
    groups: list[tuple[str, Sequence[str]]] = _populated_groups(frozenset(names))
    if groups:
        return _grouped_variables_cell(names, groups)

    ordered: tuple[int, ...] = display_order(names)
    return ", ".join(_symbol(names[i]) for i in ordered)


def _gaussian_params_cell(params: Mapping[str, Any], /) -> str:
    """A Gaussian run's `dim`/`mu_*`/`cov_*` as text.

    There is no scalar slot for a mean vector or a covariance matrix, so the whole dict
    is rendered `key=value` and spans the row like `variables` does for a jet run.
    """
    rendered: str = ", ".join(f"{k}={_format_value(v)}" for k, v in params.items())
    return rf"\ConfigVal{{{rendered}}}"


def _sigma_cell(sigmas: Sequence[float], /) -> str:
    r"""`median x (1/2 .. 2)` when the values really are the bracket else raw."""
    # Deferred: `deconvolve.training.mmd` imports jax, which this module must
    # not load eagerly.
    from ..training.mmd import _SCALES

    if len(sigmas) == len(_SCALES):
        median: float = sigmas[_SCALES.index(1.0)]
        if all(
            np.isclose(a=s, b=median * scale, rtol=1e-6)
            for s, scale in zip(sigmas, _SCALES, strict=True)
        ):
            return (
                f"{_num('Raw', median)} "
                r"\( \times\ (\frac12,\ \frac1{\sqrt2},\ 1,\ \sqrt2,\ 2)\)"
            )
    return ", ".join(_num("Raw", s) for s in sigmas)


def _pair_lines(scalars: list[tuple[str, Any]], /) -> list[str]:
    r"""`\ConfigPair` rows, two entries per line, the trailing odd one padded."""
    lines: list[str] = []
    for i in range(0, len(scalars), 2):
        pair: list[tuple[str, Any]] = scalars[i : i + 2]
        if len(pair) == 1:
            pair.append(("", ""))
        (k1, v1), (k2, v2) = pair
        c1: str = "" if k1 == "" else _scalar_cell(v1)
        c2: str = "" if k2 == "" else _scalar_cell(v2)
        lines.append(rf"\ConfigPair{{{k1}}}{{{c1}}}{{{k2}}}{{{c2}}}")
    return lines


def _wide_lines(entries: Mapping[str, Any], /) -> list[str]:
    r"""`\ConfigWide` rows: the variable list, Gaussian params, MMD sigmas."""
    lines: list[str] = []
    if "variables" in entries:
        # Symbols, since the cell is math, not a detokenized identifier list.
        lines.append(
            rf"\ConfigWide{{variables}}{{{_variables_cell(entries['variables'])}}}"
        )
    if "gaussian_params" in entries:
        cell: str = _gaussian_params_cell(entries["gaussian_params"])
        lines.append(rf"\ConfigWide{{gaussian_params}}{{{cell}}}")
    lines.extend(
        rf"\ConfigWide{{{key}}}{{{_sigma_cell(entries[key])}}}"
        for key in ("mmd_sigmas_detector", "mmd_sigmas_particle")
        if key in entries
    )
    return lines


def config_rows(config: Mapping[str, Any], timings: Mapping[str, Any] | None, /) -> str:
    """`<<CONFIG_ROWS>>`: two key/value pairs per line, wide values spanning.

    `compile_cache_warm` lives at the top level of `timings.json`, not in `config.json`.
    It is folded in as a config row because it makes the `compile` timing interpretable.
    """
    entries: dict[str, Any] = dict(config)
    if timings is not None and "compile_cache_warm" in timings:
        entries["compile_cache_warm"] = timings["compile_cache_warm"]

    scalars: list[tuple[str, Any]] = [
        (k, v) for k, v in entries.items() if k not in _WIDE_KEYS
    ]
    return "\n".join(_pair_lines(scalars) + _wide_lines(entries))


def _phase_line(phase: Mapping[str, Any], total: float, /) -> str:
    """One timing-table row for a single recorded phase."""
    name: str = latex_text(phase["name"])
    depth: int = phase["depth"]
    seconds: float = float(phase["seconds"])
    if depth:
        name = rf"\TimingSubphase{{{name}}}"
    # A nested phase is already inside its parent's share, so its cell is empty
    share: str = decimal(100.0 * seconds / total) if depth == 0 and total > 0 else ""
    # `pass` ("train"/"load") has no column of its own; it is folded into Detail.
    parts: list[str] = [p for p in (phase.get("detail"), phase.get("pass")) if p]
    detail: str = latex_text(" -- ".join(parts))
    if phase.get("failed"):
        detail = rf"{detail} \textbf{{(raised)}}" if detail else r"\textbf{(raised)}"
    return f"{name} & {decimal(seconds)} & {share} & {detail} \\\\"


def timing_rows(timings: Mapping[str, Any], /) -> str:
    """`<<TIMINGS_ROWS>>`: one row per phase, a rule, then the bold total."""
    total: float = float(timings["total_seconds"])
    lines: list[str] = [_phase_line(phase, total) for phase in timings["phases"]]
    lines.extend((r"\midrule", rf"\textbf{{total}} & {decimal(total)} & 100 & \\"))
    return "\n".join(lines)


# Column groups, in the order the report presents them.
_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("WASSERSTEIN", "wasserstein"),
    ("JS", "jensenshannon"),
    ("VLC", "triangular"),
)

# Observable, Sim, then a (value, improvement) pair for each of IBU, OmniFold and RAN.
# Three separate `\multicolumn` spans and the template's column spec have to agree.
_TABLE_COLUMNS: Final[int] = 8


def _entry(
    source: Mapping[str, Any] | None, level: str, variable: str | None, /
) -> Mapping[str, Any] | None:
    """One method's record for a variable at a level, or `None` where it has none.

    `variable=None` is the level's joint record, which lives one level down
    under `JOINT_METRICS_KEY` rather than at `<level>_<variable>`. A metrics
    file written before the joint metrics existed simply has none.
    """
    if source is None:
        return None
    if variable is None:
        joint: Mapping[str, Any] = source.get(JOINT_METRICS_KEY) or {}
        return joint.get(level)
    return source.get(f"{level}_{variable}")


def _metric_value(entry: Mapping[str, Any] | None, metric_key: str, /) -> float | None:
    """Extract a finite metric value from one record, or `None`."""
    if entry is None:
        return None
    val: object = entry.get(metric_key)
    if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    return None


def _best_methods(
    variable: str | None,
    level: str,
    metric: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    omnifold: Mapping[str, Any] | None,
    daggered: bool,
    /,
) -> frozenset[str]:
    """Identify which method(s) achieved the best (lowest) distance after unfolding.

    IBU is excluded if purity binning failed (`daggered`). Only evaluated when at least
    two methods are competing.
    """
    metric_key: str = f"{metric}_after"
    sources: tuple[tuple[str, Mapping[str, Any] | None], ...] = (
        ("ran", ran),
        ("ibu", None if daggered else ibu),
        ("omnifold", omnifold),
    )
    candidates: dict[str, float] = {
        name: val
        for name, src in sources
        if (val := _metric_value(_entry(src, level, variable), metric_key)) is not None
    }
    if len(candidates) < 2:
        return frozenset()

    best_val: float = min(candidates.values())
    return frozenset(
        m
        for m, v in candidates.items()
        if np.isclose(a=v, b=best_val, rtol=1e-7, atol=1e-12)
    )


def _method_cells(
    source: Mapping[str, Any] | None,
    level: str,
    variable: str | None,
    metric: str,
    /,
    *,
    is_best: bool = False,
) -> tuple[str, str]:
    """One method's `(value, improvement)` pair, or dashes where it has not run.

    A baseline that has not been run on a directory has no entry, and the template fixes
    the column count, so the pair has to be *filled* rather than omitted.
    """
    entry: Mapping[str, Any] | None = _entry(source, level, variable)
    if entry is None:
        return (_DASH, _DASH)
    val: str = decimal(float(entry[f"{metric}_after"]) * METRIC_DISPLAY_SCALE)
    impr: str = decimal(entry[f"{metric}_improvement_pct"])
    if is_best:
        if val != _DASH:
            val = rf"\bfseries {val}"
        if impr != _DASH:
            impr = rf"\bfseries {impr}"
    return (val, impr)


# The joint table's single row stands for every observable at once; kept no wider
# than the "Observable" header, so its columns line up with the tables above it.
_JOINT_LABEL: Final[str] = r"\textit{All}"


def _row(
    variable: str | None,
    level: str,
    metric: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    omnifold: Mapping[str, Any] | None,
    daggered: bool,
    /,
) -> str:
    """One variable's eight cells: label, Sim, then a pair per method.

    RAN goes last: the eye reads a row left to right and stops at the end, so the method
    under test sits where a reader lands, with the baselines in front of it.
    `variable=None` is the level's joint row, which RAN's file must carry.
    """
    symbol: str = _JOINT_LABEL if variable is None else _symbol(variable)
    label: str = rf"{symbol}\(^\dag\)" if daggered else symbol
    ours: Mapping[str, Any] = cast(
        typ="Mapping[str, Any]", val=_entry(ran, level, variable)
    )

    best: frozenset[str] = _best_methods(
        variable, level, metric, ran, ibu, omnifold, daggered
    )

    cells: list[str] = [
        label,
        decimal(float(ours[f"{metric}_before"]) * METRIC_DISPLAY_SCALE),
    ]
    cells.extend(_method_cells(ibu, level, variable, metric, is_best="ibu" in best))
    cells.extend(
        _method_cells(omnifold, level, variable, metric, is_best="omnifold" in best)
    )
    cells.extend(_method_cells(ran, level, variable, metric, is_best="ran" in best))
    return " & ".join(cells) + r" \\"


_DAGGER_LEGEND: Final[str] = (
    rf"\rowcolor{{white}}\multicolumn{{{_TABLE_COLUMNS}}}{{@{{}}l}}"
    r"{\footnotesize \(^\dag\) IBU's purity binning produced a single bin, "
    r"so IBU failed to unfold.} \\"
)


def metrics_table(
    level: str,
    metric: str,
    variables: Sequence[str],
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    omnifold: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
    *,
    include_legend: bool = True,
) -> str:
    """One level's row bodies for one metric.

    The template owns the tabular, column spec and header; this owns the initial rule,
    data rows in physics display order, and optional dagger legend.
    """
    ordered: tuple[int, ...] = display_order(variables)
    ordered_vars: list[str] = [variables[i] for i in ordered]
    rows: list[str] = [
        _row(v, level, metric, ran, ibu, omnifold, v in skipped) for v in ordered_vars
    ]
    lines: list[str] = [r"\midrule", *rows]

    if include_legend and any(v in skipped for v in ordered_vars):
        lines.append(_DAGGER_LEGEND)
    return "\n".join(lines)


# A `metrics.json` from before the joint metrics: one spanning row naming the
# fix, rather than a row of dashes that reads as "every method failed".
_NO_JOINT: Final[str] = (
    r"\midrule"
    "\n"
    rf"\multicolumn{{{_TABLE_COLUMNS}}}{{@{{}}l}}{{\itshape No joint metrics in "
    r"metrics.json: rerun \texttt{deconvolve evaluate --force}.} \\"
)


def joint_table(
    level: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    omnifold: Mapping[str, Any] | None,
    /,
) -> str:
    """One level's sliced Wasserstein body: a single row over every observable.

    Same eight columns as the per-variable tables, so the template reuses their
    spec and header. IBU is never daggered here: its joint score is taken on
    the product of its per-variable weights, in which a refused variable is
    simply a factor of one, so there is always a number to show.
    """
    if _entry(ran, level, None) is None:
        return _NO_JOINT
    row: str = _row(None, level, "sliced_wasserstein", ran, ibu, omnifold, False)
    return f"\\midrule\n{row}"


def skipped_variables(
    run_dir: Path, ibu: Mapping[str, Any] | None, /
) -> frozenset[str]:
    """Variables IBU's purity binning refused, from the recorded outcomes."""
    path: Path = run_dir / ARTIFACTS_DIR / "ibu_outcomes.json"
    try:
        outcomes: list[dict[str, Any]] = json.loads(s=path.read_text())
    except (OSError, ValueError):
        if ibu is None:
            return frozenset()
        return frozenset(
            key.split(sep="_", maxsplit=1)[1]
            for key, entry in ibu.items()
            if key != JOINT_METRICS_KEY
            and entry["wasserstein_after"] == entry["wasserstein_before"]
        )
    return frozenset(o["variable_name"] for o in outcomes if o["status"] == "skipped")


# A run that died before `deconvolve evaluate`: the tables degrade to a single
# explanatory row rather than raising. The template fixes column count, so row
# has to span all of them.
_NO_METRICS: Final[str] = (
    r"\midrule"
    "\n"
    rf"\multicolumn{{{_TABLE_COLUMNS}}}{{@{{}}l}}{{\itshape metrics.json not "
    r"found: run \texttt{ran evaluate} for this run.} \\"
)


def _read(path: Path, /) -> dict[str, Any] | None:
    """One JSON artifact, or `None` when it is absent or unreadable."""
    try:
        return cast(
            typ="dict[str, Any]", val=json.loads(s=path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError):
        return None


def _variables(config: Mapping[str, Any], /) -> tuple[str, ...]:
    """The run's column names: recorded for a jet run, positional otherwise."""
    recorded: object = config.get("variables")
    if recorded:
        return tuple(cast(typ="Sequence[str]", val=recorded))
    return tuple(f"dim_{i}" for i in range(int(config["dim"])))


def _table(
    level: str,
    metric: str,
    variables: Sequence[str],
    ran: Mapping[str, Any] | None,
    ibu: Mapping[str, Any] | None,
    omnifold: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
    *,
    include_legend: bool = False,
) -> str:
    """A metrics body, or the not-found row when there are no metrics."""
    if not ran:
        return _NO_METRICS
    return metrics_table(
        level,
        metric,
        variables,
        ran,
        ibu,
        omnifold,
        skipped,
        include_legend=include_legend,
    )


def _joint(
    level: str,
    ran: Mapping[str, Any] | None,
    ibu: Mapping[str, Any] | None,
    omnifold: Mapping[str, Any] | None,
    /,
) -> str:
    """A joint body, or the not-found row when there are no metrics at all."""
    if not ran:
        return _NO_METRICS
    return joint_table(level, ran, ibu, omnifold)


def _figure_pages(artifacts: Path, stem: str, dim: int, /) -> str:
    r"""One `\ReportGraphic` block per page of a paginated level figure.

    `plotting._plot_level` writes the pages of one multi-page PDF, six panels
    each. The count is `figure_pages(dim)` rather than something read off the
    file, which keeps `report.py` free of a PDF dependency and free of
    matplotlib -- the two agree because they share the constant.
    """
    path: str = str(object=artifacts.resolve() / f"{stem}.pdf")
    return "\n\\clearpage\n".join(
        rf"\ReportPage{{{path}}}{{{page}}}" for page in range(1, figure_pages(dim) + 1)
    )


def render(run_dir: Path, /) -> str:
    """The fully substituted LaTeX source for one run directory."""
    # There is no run without a config
    config_path: Path = run_dir / "config.json"
    config: dict[str, Any] | None = _read(config_path)
    if config is None:
        state: str = "is unreadable" if config_path.exists() else "does not exist"
        msg: str = f"{config_path} {state}: not a run directory"
        raise FileNotFoundError(msg)

    artifacts: Path = run_dir / ARTIFACTS_DIR
    ran: dict[str, Any] | None = _read(artifacts / "metrics.json")
    ibu: dict[str, Any] | None = _read(artifacts / "metrics_ibu.json")
    omnifold: dict[str, Any] | None = _read(artifacts / "metrics_omnifold.json")
    timings: dict[str, Any] | None = _read(artifacts / "timings.json")
    skipped: frozenset[str] = skipped_variables(run_dir, ibu)
    variables: tuple[str, ...] = _variables(config)
    has_particle: bool = bool(ran and any(k.startswith("particle_") for k in ran))
    legend_level: str = "particle" if has_particle else "detector"

    source: str = load_template()
    for token, value in (
        ("<<RUN_NAME>>", run_dir.name),
        ("<<CONFIG_ROWS>>", config_rows(config, timings)),
        ("<<TIMINGS_ROWS>>", timing_rows(timings) if timings else ""),
        *(
            (
                f"<<{level.upper()}_{metric_tag}>>",
                _table(
                    level,
                    metric,
                    variables,
                    ran,
                    ibu,
                    omnifold,
                    skipped,
                    include_legend=(level == legend_level and metric_tag == "VLC"),
                ),
            )
            for level in ("detector", "particle")
            for metric_tag, metric in _METRICS
        ),
        *(
            (f"<<{level.upper()}_SLICED>>", _joint(level, ran, ibu, omnifold))
            for level in ("detector", "particle")
        ),
        (
            "<<DETECTOR_FIGURES>>",
            _figure_pages(artifacts, "detector_level", len(variables)),
        ),
        (
            "<<PARTICLE_FIGURES>>",
            _figure_pages(artifacts, "particle_level", len(variables)),
        ),
        # Absolute: `pdflatex` runs in `artifacts/`, so a relative path would
        # not resolve, and a sweep arm's directory name (`lrg1e-4_seed03`)
        # cannot be reconstructed from a bare basename either.
        ("<<FIGURE_DIR>>", str(object=artifacts.resolve())),
    ):
        source = source.replace(token, value)

    left: list[str] = TEMPLATE_TOKEN.findall(string=source)
    if left:
        msg = f"template tokens with no value: {', '.join(sorted(set(left)))}"
        raise ValueError(msg)
    return source


_LATEX_ARGS: Final[tuple[str, ...]] = (
    "pdflatex",
    "-interaction=nonstopmode",
    "-halt-on-error",
)
# Kept on failure so the compile can be debugged; removed on success
_AUX_SUFFIXES: Final[tuple[str, ...]] = (".aux", ".log", ".out")


def _compile(source: Path, artifacts: Path, run_dir: Path, /) -> None:
    r"""Run `pdflatex` twice, from `artifacts/`, emitting into the run root."""
    if shutil.which("pdflatex") is None:
        msg: str = (
            "pdflatex is not on PATH. Install a TeX distribution, or pass "
            "--no-compile to emit report.tex alone."
        )
        raise RuntimeError(msg)

    for _pass in range(2):
        completed: subprocess.CompletedProcess[str] = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            args=[*_LATEX_ARGS, f"-output-directory={run_dir}", source.name],
            cwd=artifacts,
            capture_output=True,
            text=True,
            check=False,
            # `-interaction=nonstopmode` is set inside the document processor
            # and does not cover pdflatex's *pre-mode* prompts
            stdin=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            tail: str = "\n".join(completed.stdout.splitlines()[-40:])
            msg = f"pdflatex failed for {run_dir.name}:\n{tail}"
            raise RuntimeError(msg)


def build_report(
    run_dir: Path, /, *, force: bool = False, compile_pdf: bool = True
) -> Path:
    """Write `artifacts/report.tex`, compile `report.pdf` at the run root.

    Returns: the PDF, or the LaTeX source under `--no-compile`.
    """
    run_dir: Path = run_dir.resolve()
    pdf: Path = run_dir / "report.pdf"
    if compile_pdf and pdf.exists() and not force:
        logger.info("%s: report.pdf exists, skipping (use --force)", run_dir.name)
        return pdf

    source: str = render(run_dir)
    tex: Path = artifacts_dir(run_dir) / "report.tex"
    _ = tex.write_text(data=source, encoding="utf-8")
    if not compile_pdf:
        logger.info("%s: saved %s", run_dir.name, tex)
        return tex

    _compile(tex, tex.parent, run_dir)
    for suffix in _AUX_SUFFIXES:
        (run_dir / f"report{suffix}").unlink(missing_ok=True)
    logger.info("%s: saved %s", run_dir.name, pdf)
    return pdf
