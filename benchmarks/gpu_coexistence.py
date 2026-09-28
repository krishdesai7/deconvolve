"""Can a TensorFlow subprocess get GPU memory while a JAX parent holds the card?

This is the fail-fast check standing in front of any decision to bring OmniFold
into this repository. The quarantine itself is settled -- a PEP 723 script run
through `uv run --no-project` provisions Python 3.13 and TensorFlow in an
interpreter that cannot see `deconvolve`, which is proven and cheap. What is *not*
settled is whether the two processes can share one A100, and that question only
has an answer on a machine with an A100 in it.

The hazard is in the Precision section of CLAUDE.md: **JAX preallocates ~75% of
GPU memory on its first device allocation.** A `deconvolve baseline omnifold` parent
reaches the device long before it spawns a worker -- `load_populations` alone
does it -- so by the time TensorFlow starts, three quarters of the card is
already spoken for and does not come back. Deleting the array does not release
it; the preallocation is the pool, not the array, which is why there is no
"just free it first" arm below.

Five arms, each a separate process because `XLA_PYTHON_CLIENT_*` is read when
the JAX backend initialises and cannot be changed afterwards. Reading them in
one interpreter and re-importing is the obvious way to write this and it
measures nothing: every arm after the first would inherit the first one's pool.

| Arm                   | Parent                    | What a pass means           |
| --------------------- | ------------------------- | --------------------------- |
| `control`             | never touches JAX         | the worker works on this node |
| `preallocate-default` | JAX on GPU, defaults      | no fix needed               |
| `preallocate-false`   | `..._PREALLOCATE=false`   | the cheap fix works         |
| `mem-fraction-0.4`    | `..._MEM_FRACTION=0.4`    | the budgeted fix works      |
| `parent-on-cpu`       | `JAX_PLATFORMS=cpu`       | the fallback design works   |

(the two truncated names are `XLA_PYTHON_CLIENT_PREALLOCATE` and
`XLA_PYTHON_CLIENT_MEM_FRACTION`.)

`control` is the arm to read first. If it fails, nothing below it means
anything -- the node, the CUDA driver or the uv script cache is the problem,
not coexistence -- and the other four results should be discarded rather than
interpreted.

Run it inside a GPU allocation, not on a login node:

    srun -C gpu --qos=shared --gpus=1 --cpus-per-task=32 --time=00:20:00 \
        uv run benchmarks/gpu_coexistence.py

**Warm the worker environment on a login node first.** uv resolves the PEP 723
header at first run and compute nodes generally have no outbound network, so a
cold cache fails the whole benchmark with a download error that looks nothing
like an OOM:

    uv run --no-project benchmarks/_tf_probe_worker.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.table import Table

WORKER: Path = Path(__file__).parent / "_tf_probe_worker.py"

# Large enough to force a real device allocation, small enough that it is never
# itself the reason the card is full: what fills the card is the preallocation,
# and the point of the benchmark is to attribute the failure to that.
PARENT_ARRAY_BYTES: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class Arm:
    name: str
    touch_jax: bool
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


ARMS: tuple[Arm, ...] = (
    Arm("control", touch_jax=False, note="worker alone on the card"),
    Arm(
        "preallocate-default",
        touch_jax=True,
        note="what `deconvolve baseline` would do today",
    ),
    Arm(
        "preallocate-false",
        touch_jax=True,
        env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        note="JAX grows on demand",
    ),
    Arm(
        "mem-fraction-0.4",
        touch_jax=True,
        env={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4"},
        note="JAX capped, ~60% left",
    ),
    Arm(
        "parent-on-cpu",
        touch_jax=True,
        env={"JAX_PLATFORMS": "cpu"},
        note="parent never on device",
    ),
)


def nvidia_free_mib() -> tuple[int, int] | None:
    """`(free, total)` MiB on GPU 0, or None where there is no nvidia-smi."""
    # Fixed argv, no shell. `nvidia-smi` is deliberately a bare name: it is
    # resolved on PATH because its location differs between the driver packages
    # a cluster might have installed.
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    try:
        free, total = (int(v.strip()) for v in out.split(","))
    except ValueError:
        return None
    return free, total


# Substrings of the TF/absl log lines that actually bear on GPU discovery. The
# raw stderr is thousands of lines of op-registration noise at log level 0, and
# dumping it buries the three lines that matter.
# TensorFlow names the library it gave up on in exactly this form, at VLOG(1)
# from dso_loader. This is the authoritative answer -- see `_dump_dlopen` for
# why the worker's own dlopen attempt is not.
_MISSING_RE: re.Pattern[str] = re.compile(r"Could not load dynamic library '([^']+)'")


def _missing_libraries(stderr: str) -> list[str]:
    """The sonames TensorFlow itself reported it could not open."""
    return sorted(set(_MISSING_RE.findall(stderr)))


_CUDA_SIGNALS: tuple[str, ...] = (
    "cuInit",
    "cuda",
    "CUDA",
    "cudnn",
    "cuDNN",
    "cublas",
    "libcu",
    "GPU",
    "gpu_device",
    "numa",
    "dynamic library",
    "StreamExecutor",
)


def _cuda_lines(stderr: str, limit: int = 40) -> list[str]:
    """The lines of TF's stderr that say something about finding a GPU."""
    hits = [
        line.strip()
        for line in stderr.splitlines()
        if any(signal in line for signal in _CUDA_SIGNALS)
    ]
    # Deduplicated because TF repeats the same warning once per registered op.
    seen: dict[str, None] = {}
    for line in hits:
        seen.setdefault(line, None)
    return list(seen)[:limit]


def run_worker() -> dict[str, object]:
    """Spawn the TensorFlow worker and parse its one line of JSON.

    `--no-project` is not decoration. Without it uv would resolve the script
    against this repository's `pyproject.toml` and run it in the project
    environment -- the one environment that must never hold TensorFlow, and
    whose interpreter is not the `==3.13.*` the worker pins. That fails loudly
    rather than silently, which is the good case, but it fails.
    """
    # Fixed argv, no shell; the only interpolated element is a path inside
    # this file's own directory. `uv` is a bare name on purpose -- it is what
    # the user invoked this benchmark with.
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["uv", "run", "--no-project", str(WORKER)],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    line = proc.stdout.strip().splitlines()
    if not line:
        return {
            "status": "error",
            "detail": f"worker exited {proc.returncode} with no JSON",
            "stderr": proc.stderr[-1500:],
        }
    try:
        parsed: dict[str, object] = json.loads(line[-1])
    except json.JSONDecodeError:
        return {
            "status": "error",
            "detail": "unparseable worker stdout",
            "stdout": proc.stdout[-800:],
            "stderr": proc.stderr[-1500:],
        }
    # TF writes its CUDA diagnostics to stderr and its verdict to stdout, so a
    # result without the stderr attached cannot explain itself.
    if parsed.get("status") != "ok":
        parsed["stderr_tail"] = _cuda_lines(proc.stderr)
        # The filtered view is what gets printed, but the filter is a guess at
        # which lines matter and a missed line costs another GPU allocation to
        # recover. The unfiltered tail goes to `--json` and is never printed.
        parsed["stderr_full"] = proc.stderr[-40000:]
        parsed["missing_libraries"] = _missing_libraries(proc.stderr)
    return parsed


def as_parent(arm: Arm) -> dict[str, object]:
    """The middle process: hold the card the way a real run would, then spawn.

    The array is bound to a local and read after the worker returns so that
    nothing -- not the garbage collector, not a clever XLA rewrite -- can retire
    it while the worker is running. A parent that quietly released the device
    would report a pass that the real baseline could not reproduce.
    """
    report: dict[str, object] = {"arm": arm.name}
    held = None

    if arm.touch_jax:
        import jax
        import jax.numpy as jnp

        n = PARENT_ARRAY_BYTES // 4
        held = jnp.ones((n,), dtype=jnp.float32)
        jax.block_until_ready(held)
        report["parent_platform"] = jax.default_backend()
        report["parent_device"] = str(held.device)
    else:
        report["parent_platform"] = "none"
        report["parent_device"] = "none"

    mem = nvidia_free_mib()
    if mem is not None:
        report["free_mib_before_worker"], report["total_mib"] = mem

    report["worker"] = run_worker()

    if held is not None:
        # Reading the array after the worker returns is what forces the parent
        # to still own it for the worker's whole lifetime. The value is checked
        # loosely -- the question is whether the buffer survived, not arithmetic.
        report["parent_array_live"] = abs(float(held[0]) - 1.0) < 1e-6
    return report


def _worker_status(row: dict[str, object]) -> str:
    """A row's worker status, narrowed from the JSON it arrived as.

    A blanket `# type: ignore` on a chained `.get()` was what this used to be,
    and it silenced the checkers without telling either of them what the shape
    is. A row whose `worker` key is missing or malformed is an error, which is
    the same thing a crashed worker reports.
    """
    worker = row.get("worker")
    if not isinstance(worker, dict):
        return "error"
    return str(cast("dict[str, object]", worker).get("status", "error"))


def verdict(row: dict[str, object]) -> tuple[str, str]:
    status = _worker_status(row)
    return {
        "ok": ("[green]PASS[/green]", "worker deconvolve on the GPU"),
        "oom": ("[red]FAIL[/red]", "worker could not get memory"),
        "cpu_fallback": (
            "[yellow]SILENT[/yellow]",
            "worker saw no GPU and used the CPU",
        ),
    }.get(status, ("[red]ERROR[/red]", "worker did not report"))


def render(rows: list[dict[str, object]], console: Console) -> None:
    table = Table(title="TensorFlow worker vs. a JAX parent, one GPU", show_lines=False)
    for col in ("arm", "parent", "free MiB", "worker", "result", "what it means"):
        table.add_column(col, overflow="fold")
    for row in rows:
        mark, meaning = verdict(row)
        free = row.get("free_mib_before_worker")
        total = row.get("total_mib")
        table.add_row(
            str(row.get("arm")),
            str(row.get("parent_platform")),
            f"{free} / {total}" if free is not None else "-",
            _worker_status(row),
            mark,
            meaning,
        )
    console.print(table)


def _as_str_list(value: object) -> list[str]:
    """A JSON-crossing list, narrowed. Element types do not survive the trip."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in cast("list[object]", value)]


def _dump_dlopen(
    report: dict[str, object], wheel_libs: list[str], console: Console
) -> None:
    """Which sonames resolve on the *system* search path -- not TF's.

    Read this as supporting evidence and never as the verdict. A bare
    `ctypes.CDLL("libcublas.so.12")` searches `LD_LIBRARY_PATH` and the ldconfig
    cache; TensorFlow additionally reaches its own pip wheel directories
    through the RPATH baked into its extension modules. So this table
    over-reports: measured on Perlmutter it called nine libraries unreachable
    while TF's own log showed it had opened eight of them and failed on exactly
    one. Acting on this column alone fixes the wrong thing.

    What it does establish is the difference between a library that is *absent*
    and one that is merely *off the system path*, which is why each row is
    cross-referenced against the wheels on disk.
    """
    table = Table(title="system search path only -- NOT TensorFlow's (see note)")
    table.add_column("library", overflow="fold")
    table.add_column("ctypes.CDLL", overflow="fold")
    table.add_column("in a pip wheel?", overflow="fold")

    def wheel_note(name: str) -> str:
        return "[yellow]yes[/yellow]" if name in wheel_libs else "[red]no[/red]"

    failures = {k: v for k, v in report.items() if str(v) != "ok"}
    for name, detail in failures.items():
        table.add_row(f"[red]{name}[/red]", str(detail), wheel_note(name))
    for name in report:
        if name not in failures:
            table.add_row(name, "[green]ok[/green]", wheel_note(name))
    console.print(table)
    if failures:
        console.print(
            f"{len(failures)} of {len(report)} did not resolve on the system "
            "path. Rows marked [yellow]yes[/yellow] are present in a pip wheel "
            "and reachable by TF anyway -- compare `missing_libraries`, which "
            "is TF's own verdict."
        )


def _dump_environment(env: dict[str, object], console: Console) -> None:
    """The worker's environment, with the dlopen results broken out."""
    env_map = dict(env)
    # Pulled out and rendered separately: it is the answer, and folded into a
    # single cell of the environment table it is unreadable.
    dlopen = env_map.pop("dlopen", None)

    table = Table(title="worker environment", show_header=False)
    table.add_column("key", style="cyan", overflow="fold")
    table.add_column("value", overflow="fold")
    for key, value in env_map.items():
        table.add_row(key, str(value))
    console.print(table)

    if isinstance(dlopen, dict):
        raw_wheels = env_map.get("nvidia_wheel_libs")
        wheels = _as_str_list(raw_wheels)
        _dump_dlopen(cast("dict[str, object]", dlopen), wheels, console)


def _dump_tf_verdict(worker: dict[str, object], console: Console) -> None:
    """TensorFlow's own account: the libraries it named, then its CUDA log."""
    missing = _as_str_list(worker.get("missing_libraries"))
    if missing:
        names = ", ".join(missing)
        console.print(
            f"[bold red]TensorFlow could not open: {names}[/bold red]\n"
            "That is TF's own verdict and the one to act on. One unreachable "
            "library is enough for it to skip every GPU."
        )

    lines = worker.get("stderr_tail")
    if isinstance(lines, list) and lines:
        console.print("[bold]TensorFlow's CUDA log lines:[/bold]")
        for line in _as_str_list(lines):
            console.print(f"  {line}")
    elif isinstance(lines, list):
        console.print(
            "[yellow]TensorFlow logged nothing about CUDA at all[/yellow] -- "
            "that points at a CPU-only wheel rather than a driver problem; "
            "check `built_with_cuda` above."
        )


def _dump_worker_diagnosis(row: dict[str, object], console: Console) -> None:
    """Print the worker's own account of what it found, and what it did not."""
    worker = row.get("worker")
    if not isinstance(worker, dict):
        return

    env = worker.get("environment")
    if isinstance(env, dict):
        _dump_environment(cast("dict[str, object]", env), console)

    _dump_tf_verdict(worker, console)


def _explain(
    rows: list[dict[str, object]],
    selected: list[Arm],
    *,
    has_cuda: bool,
    console: Console,
) -> None:
    """Say what the table means, distinguishing "wrong machine" from "broken".

    A run on a laptop produces five identical `cpu_fallback` rows, which is the
    correct answer to a question the laptop cannot be asked. Reporting that as
    a failure of the node or the driver sends the reader debugging a machine
    that was never a candidate, so the absence of `nvidia-smi` is checked
    before anything is blamed on it.
    """
    if not has_cuda:
        console.print(
            "[yellow]No CUDA GPU on this machine[/yellow] -- `nvidia-smi` is "
            "absent, so every arm falls back to the CPU and the table says "
            "nothing about coexistence. The plumbing above is verified; run "
            "this inside a GPU allocation for the actual answer."
        )
        return
    if (
        rows
        and selected[0].name == "control"
        and verdict(rows[0])[0] != "[green]PASS[/green]"
    ):
        console.print(
            "[red]control did not pass[/red] -- the node, the CUDA driver or "
            "the uv script cache is the problem, not coexistence. Discard the "
            "other four arms rather than interpreting them."
        )
        _dump_worker_diagnosis(rows[0], console)


def _parent_row(arm: Arm, proc: subprocess.CompletedProcess[str]) -> dict[str, object]:
    """One table row from a finished parent process, or an account of its death."""
    out = proc.stdout.strip().splitlines()
    if out:
        parent_report: dict[str, object] = json.loads(out[-1])
        return parent_report | {"env": arm.env}
    return {
        "arm": arm.name,
        "env": arm.env,
        "parent_platform": "?",
        "worker": {
            "status": "error",
            "detail": f"parent exited {proc.returncode}",
        },
        "stderr": proc.stderr[-1500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ = parser.add_argument(
        "--arm",
        action="append",
        choices=[a.name for a in ARMS],
        help="run only these arms (repeatable); default is all five",
    )
    _ = parser.add_argument("--json", type=Path, help="also write the raw results here")
    _ = parser.add_argument(
        "--preload-wheels",
        action="store_true",
        help="prepend the pip CUDA wheel directories to LD_LIBRARY_PATH in the "
        "worker, to test whether a system CUDA is shadowing them",
    )
    _ = parser.add_argument("--as-parent", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Set before any arm runs so it reaches the worker through both
    # subprocess layers, which inherit the environment.
    if args.preload_wheels:
        os.environ["DECONVOLVE_PROBE_PRELOAD_WHEELS"] = "1"

    by_name = {a.name: a for a in ARMS}

    # Re-exec of self as the middle process. Reached only via subprocess below.
    if args.as_parent:
        print(json.dumps(as_parent(by_name[args.as_parent])))
        return

    console = Console(stderr=True)
    if not WORKER.exists():
        console.print(f"[red]missing worker script:[/red] {WORKER}")
        raise SystemExit(2)

    selected = [by_name[n] for n in (args.arm or [a.name for a in ARMS])]
    rows: list[dict[str, object]] = []

    for arm in selected:
        console.print(f"[cyan]arm[/cyan] {arm.name} -- {arm.note}")
        env = os.environ | arm.env
        # Fixed argv, no shell: this interpreter, this file, a name from ARMS.
        proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, __file__, "--as-parent", arm.name],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=3600,
        )
        rows.append(_parent_row(arm, proc))

    render(rows, console)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        console.print(f"wrote {args.json}")

    _explain(rows, selected, has_cuda=nvidia_free_mib() is not None, console=console)


if __name__ == "__main__":
    main()
