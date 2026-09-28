from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

import numpy as np
import typer
from typer.core import TyperGroup, TyperOption

from .baselines import ibu_evaluate_runs, omnifold_evaluate_runs
from .config import ConfigError, default_map, discover, load, origins_for
from .config.show import render
from .config.spec import build_spec
from .coretypes import (
    DEFAULT_PURITY_THRESHOLD,
    POISON_SENTINEL,
    SUBSTRUCTURE_VARIABLES,
    DatasetName,
    LogLevel,
    Resample,
)
from .evaluation import evaluate_runs
from .instrumentation import configure_logging
from .reporting import build_report
from .uncertainty import DesignSpec, collect, freeze_design, load_frozen, run_cell
from .workflows import run, run_leakage_check

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any, Final, LiteralString

    from typer._click.core import Command, Context, ParameterSource

    from .config import Resolved
    from .config.spec import CommandSpec

logger: Logger = logging.getLogger(name=__name__)


def _gate_autoenv(command: Command, spec: CommandSpec | None, /) -> None:
    """Give an environment variable only to options the config files may set.

    Layer 4 is the same key set as layers 2-3, not every option Click could
    derive a name for: `--force` should be typed each time, not inherited
    from a shell profile, for the same reason `NOT_LAYERABLE` keeps it out of
    `deconvolve.toml`. A command outside the spec (`FROZEN_COMMANDS`) gets no
    variables at all, so `uncertainty run --help` does not advertise
    `DECONVOLVE_UNCERTAINTY_RUN_N_EPOCHS` that `_resolve_cell_settings` would
    then ignore. (Positional arguments such as `run`'s `CELL` never read the
    environment in Typer, gated or not.) Derived from the spec rather than
    marked per option, so the denylist stays the one list to maintain. Typer's own
    `--install-completion`/`--show-completion` are in no spec either, which is
    what keeps an exported `DECONVOLVE_INSTALL_COMPLETION` from firing on
    every invocation.
    """
    # No spec means nothing is layerable, here and in every subcommand.
    options: dict[str, Any] = spec.options if spec is not None else {}
    children: dict[str, CommandSpec] = spec.children if spec is not None else {}
    for param in command.params:
        if isinstance(param, TyperOption) and param.name is not None:
            param.allow_from_autoenv = param.name in options
    if isinstance(command, TyperGroup):
        for name, child in command.commands.items():
            _gate_autoenv(child, children.get(name))


class _GatedGroup(TyperGroup):
    """The root group, gating every option's variable before anything parses.

    Not the root callback: the root's own options, completion flags included,
    are read from the environment while the root context is being built,
    before that callback runs, and `deconvolve --help` never reaches it at all.
    """

    @override
    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: Context | None = None,
        **extra: Any,
    ) -> Context:
        _gate_autoenv(self, _spec())
        return super().make_context(info_name, args, parent, **extra)


baseline_app: typer.Typer = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
uncertainty_app: typer.Typer = typer.Typer(
    rich_markup_mode="rich", no_args_is_help=True
)
config_app: typer.Typer = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)

# Layer 4. Click derives each option's variable from the command path, so
# `train --n-epochs` reads `DECONVOLVE_TRAIN_N_EPOCHS` and the root
# `--log-level` reads `DECONVOLVE_LOG_LEVEL`; `_gate_autoenv` narrows this to
# the options the config files may also set.
ENVVAR_PREFIX: Final[LiteralString] = "DECONVOLVE"

app: typer.Typer = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    cls=_GatedGroup,
    context_settings={"auto_envvar_prefix": ENVVAR_PREFIX},
)
app.add_typer(
    typer_instance=baseline_app, name="baseline", help="Run comparison baselines."
)
app.add_typer(
    typer_instance=uncertainty_app,
    name="uncertainty",
    help="Bootstrap x seed variance decomposition.",
)
app.add_typer(
    typer_instance=config_app,
    name="config",
    help="Inspect the resolved configuration stack.",
)


@cache
def _spec() -> CommandSpec:
    """Built once per process, after every command has registered itself."""
    return build_spec(app)


def _resolve_once(ctx: typer.Context, /) -> Resolved | None:
    """Resolve the config stack exactly once per invocation, and stash it.

    `train`, `uncertainty freeze` and `config show` each need the `Resolved`
    object itself, not just the merged `default_map` Click computes from it
    (to report provenance, or to render the table). They used to re-resolve
    independently, which meant a second, redundant filesystem walk and --
    worse -- a real chance that `ctx.get_parameter_source` (reflecting this
    walk) and the `Resolved` an `origins_for` lookup consulted (a *different*
    walk) disagreed about what the filesystem held, if a file appeared,
    disappeared, or an NFS attribute cache refreshed in between. Provenance
    that names the wrong file is worse than none, so every command now reads
    the single `Resolved` stashed here instead.

    A broken config file must not disable the one command whose job is to
    diagnose it, so `deconvolve config show` proceeds with no defaults and reports
    the stashed error itself. Every other command fails here.
    """
    try:
        resolved: Resolved = load(discover(Path.cwd(), os.environ), _spec())
    except ConfigError as error:
        ctx.meta["deconvolve.config_error"] = error
        if ctx.invoked_subcommand == "config":
            return None
        raise typer.BadParameter(str(object=error)) from error
    ctx.meta["deconvolve.resolved"] = resolved
    return resolved


@app.callback()
def configure(
    ctx: typer.Context,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            "-L",
            case_sensitive=False,
            help="Application log level.",
        ),
    ] = LogLevel.info,
) -> None:
    # Assigned here rather than through `Typer(context_settings=)` so that
    # discovery is lazy: it walks the filesystem once, at invocation, not at
    # import. `coretypes/constants.py` resolving `DECONVOLVE_CACHE_DIR` at import is the
    # failure mode this avoids.
    #
    # Click then applies its own precedence to what we hand it:
    #   COMMANDLINE > ENVIRONMENT > DEFAULT_MAP > DEFAULT
    # which is layers 5, 4, 3-2, 1. It also casts these values through each
    # parameter's declared type and enforces each parameter's constraints, so
    # `n-epochs = -5` from a file is rejected by the existing `min=1`.
    resolved: Resolved | None = _resolve_once(ctx)
    ctx.default_map = default_map(resolved) if resolved is not None else {}
    configure_logging(level=log_level.value)


def _canonical_variables(chosen: list[str] | None, /) -> tuple[str, ...]:
    """Fix the jet column order once, here, so nothing downstream has to guess.

    Repeated `--var` arrives as a list in whatever order it was typed. Sorting
    it into `SUBSTRUCTURE_VARIABLES` order means `--var w --var m` and
    `--var m --var w` describe the same run --- same columns, same cache key,
    same `config.json` --- rather than two runs that differ only in a permutation
    nobody chose.
    """
    if not chosen:
        return SUBSTRUCTURE_VARIABLES
    wanted: set[str] = set(chosen)
    return tuple(v for v in SUBSTRUCTURE_VARIABLES if v in wanted) + tuple(
        # Unknown names pass through in order so `load_jet_dataset` is the one
        # place that reports them, with the list of what it does accept.
        v
        for v in dict.fromkeys(chosen)
        if v not in SUBSTRUCTURE_VARIABLES
    )


@app.command(name="train")
def train_command(
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    config: Annotated[Path | None, typer.Option()] = None,
    dataset: Annotated[
        DatasetName, typer.Option("--dataset", "-D")
    ] = DatasetName.gaussian,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    load_run: Annotated[Path | None, typer.Option("--load-run", "-r")] = None,
    hidden_units: Annotated[int, typer.Option("--hidden-units", "-u", min=1)] = 64,
    n_layers: Annotated[int, typer.Option("--n-layers", "-l", min=1)] = 2,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", min=1)] = 100,
    n_disc_steps: Annotated[int, typer.Option("--n-disc-steps", "-k", min=1)] = 5,
    lr_g: Annotated[float, typer.Option(min=0.0)] = 3e-5,
    lr_d: Annotated[float, typer.Option(min=0.0)] = 1e-4,
    lambda_dispersion: Annotated[
        float,
        typer.Option(min=0.0, help="Penalty on variance of g's weights. 0 to disable."),
    ] = 0.015,
    log_every: Annotated[int, typer.Option(min=1, help="Log every N epochs.")] = 1,
    plots: Annotated[bool, typer.Option(help="Draw plots. Metrics still run.")] = True,
    run_dir: Annotated[
        Path | None,
        typer.Option(help="Where to save this run. Default: timestamp under runs/."),
    ] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    data_seed: Annotated[int, typer.Option()] = 42,
    *,
    ctx: typer.Context,
) -> None:
    resolved: Resolved = ctx.meta["deconvolve.resolved"]
    run(
        batch_size,
        n_samples,
        config,
        dataset,
        _canonical_variables(variable),
        load_run,
        hidden_units,
        n_layers,
        seed,
        data_seed,
        n_epochs=n_epochs,
        n_disc_steps=n_disc_steps,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_dispersion=lambda_dispersion,
        log_every=log_every,
        plots=plots,
        run_dir=run_dir,
        origins=origins_for(
            ctx,
            resolved,
            path=("train",),
            names=sorted(_spec().children["train"].options),
        ),
    )


@app.command(name="evaluate")
def evaluate_command(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Run, or directory of runs, to evaluate."),
    ],
    force: bool = False,
) -> None:
    evaluate_runs(run_dir, force)


@app.command(name="report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to report on.")],
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild an existing report.pdf.")
    ] = False,
    compile_pdf: Annotated[
        bool,
        typer.Option(
            "--compile/--no-compile",
            help="Compile the LaTeX, or stop at artifacts/report.tex.",
        ),
    ] = True,
) -> None:
    """Compile a run directory into one PDF dossier."""
    _ = build_report(run_dir, force=force, compile_pdf=compile_pdf)


@baseline_app.command(name="ibu")
def ibu_command(
    run_dir: Annotated[
        Path, typer.Argument(help="Run directory to add the IBU baseline to.")
    ],
    force: bool = False,
    n_iterations: Annotated[int, typer.Option("--niter", "-i", min=1)] = 10,
    purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
) -> None:
    ibu_evaluate_runs(
        run_dir,
        force,
        n_iterations,
        purity_threshold=np.double(purity_threshold),
    )


@baseline_app.command(name="omnifold")
def omnifold_command(
    run_dir: Annotated[
        Path, typer.Argument(help="Run directory to add the OmniFold baseline to.")
    ],
    force: bool = False,
    n_iterations: Annotated[int, typer.Option("--niter", "-i", min=1)] = 3,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", min=1)] = 50,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 512,
) -> None:
    """Run the OmniFold baseline in a quarantined Python 3.13 subprocess.

    Needs `uv` on PATH, and on Perlmutter `module load cudatoolkit/12.9` --
    without it TensorFlow runs on the CPU without raising.
    """
    omnifold_evaluate_runs(run_dir, force, n_iterations, n_epochs, batch_size)


@uncertainty_app.command(name="freeze")
def uncertainty_freeze_command(
    ctx: typer.Context,
    design_dir: Annotated[Path, typer.Argument(help="Design directory to freeze.")],
    force: Annotated[bool, typer.Option("--force")] = False,
    n_datasets: Annotated[int, typer.Option("--n-datasets", "-B", min=2)] = 8,
    n_seeds: Annotated[int, typer.Option("--n-seeds", "-S", min=2)] = 8,
    n_eval: Annotated[int, typer.Option(min=1)] = 100_000,
    dataset: Annotated[DatasetName, typer.Option("--dataset", "-D")] = DatasetName.jets,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    config: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    hidden_units: Annotated[int, typer.Option("--hidden-units", "-u", min=1)] = 64,
    n_layers: Annotated[int, typer.Option("--n-layers", "-l", min=1)] = 2,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", min=1)] = 100,
    n_disc_steps: Annotated[int, typer.Option("--n-disc-steps", "-k", min=1)] = 5,
    lr_g: Annotated[float, typer.Option("--lr-g", min=0.0)] = 3e-5,
    lr_d: Annotated[float, typer.Option("--lr-d", min=0.0)] = 1e-4,
    lambda_dispersion: Annotated[
        float, typer.Option("--lambda-dispersion", min=0.0)
    ] = 0.015,
    data_seed: Annotated[int, typer.Option()] = 42,
    init_seed: Annotated[int, typer.Option()] = 0,
    resample: Annotated[
        Resample,
        typer.Option(help="Which samples the bootstrap resamples: both, data or mc."),
    ] = Resample.both,
) -> None:
    """Fix the settings for a variance design before its cells are submitted.

    Run once, on the login node, between creating the design directory and
    submitting the array. Cells read this file instead of the config layers, so
    editing `deconvolve.toml` mid-array cannot split a design.
    """
    names: tuple[str, ...] = tuple(
        sorted(_spec().children["uncertainty"].children["freeze"].options)
    )
    values: dict[str, Any] = {
        "n_datasets": n_datasets,
        "n_seeds": n_seeds,
        "n_eval": n_eval,
        "dataset": dataset.value,
        "variable": list(_canonical_variables(variable)),
        "config": str(object=config) if config is not None else None,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "hidden_units": hidden_units,
        "n_layers": n_layers,
        "n_epochs": n_epochs,
        "n_disc_steps": n_disc_steps,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "lambda_dispersion": lambda_dispersion,
        "data_seed": data_seed,
        "init_seed": init_seed,
        "resample": resample.value,
    }

    resolved: Resolved = ctx.meta["deconvolve.resolved"]
    try:
        path = freeze_design(
            design_dir,
            values,
            origins_for(ctx, resolved, ("uncertainty", "freeze"), names=names),
            force=force,
        )
    except FileExistsError as error:
        raise typer.BadParameter(str(object=error)) from error
    logger.info("Froze design settings to %s", path)


def _resolve_cell_settings(
    ctx: typer.Context, frozen: dict[str, Any]
) -> dict[str, Any]:
    """The frozen design's settings, overridden only by what was typed on this line.

    An explicit flag still wins over the frozen file, but nothing else does:
    not an environment variable, and not whatever `deconvolve.toml` currently says.
    Shared by `uncertainty run` and `uncertainty collect`: the intersection of
    `frozen` and `ctx.params` naturally narrows to whichever of the frozen
    keys that command actually declares as an option, so `collect` gets back
    only `n_datasets`/`n_seeds`/`data_seed`/`init_seed`.
    """

    def _from_commandline(name: str, /) -> bool:
        source: ParameterSource | None = ctx.get_parameter_source(name)
        return source is not None and source.name == "COMMANDLINE"

    return {
        name: (ctx.params[name] if _from_commandline(name) else frozen[name])
        for name in frozen
        if name in ctx.params
    }


def _require_complete(frozen: dict[str, Any], design_dir: Path, /) -> None:
    """Fail loudly, and legibly, at cell 0 rather than obscurely at cell 63.

    A `design.json` missing a key --- truncated, hand-edited, written by an
    older `freeze` --- must not let that key silently fall back to a bare
    code default a different cell might not share.
    """
    expected: frozenset[str] = frozenset(
        _spec().children["uncertainty"].children["freeze"].options
    )
    missing: frozenset[str] = expected - frozen.keys()
    if missing:
        raise typer.BadParameter(
            f"{design_dir / 'design.json'} is missing {sorted(missing)}; rerun "
            f"`deconvolve uncertainty freeze {design_dir}` to write a "
            f"complete file"
        )


@uncertainty_app.command(name="run")
def uncertainty_run_command(
    ctx: typer.Context,
    cell: Annotated[int, typer.Argument(help="Cell number to train.", min=0)],
    design_dir: Annotated[
        Path, typer.Argument(help="Design directory to read options from.")
    ],
    n_datasets: Annotated[int | None, typer.Option("--n-datasets", "-B", min=2)] = None,
    n_seeds: Annotated[int | None, typer.Option("--n-seeds", "-S", min=2)] = None,
    n_eval: Annotated[int | None, typer.Option(min=1)] = None,
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-D")] = None,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    config: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", "-b", min=1)] = None,
    n_samples: Annotated[int | None, typer.Option("--n-samples", "-n", min=1)] = None,
    hidden_units: Annotated[
        int | None, typer.Option("--hidden-units", "-u", min=1)
    ] = None,
    n_layers: Annotated[int | None, typer.Option("--n-layers", "-l", min=1)] = None,
    n_epochs: Annotated[int | None, typer.Option("--n-epochs", "-e", min=1)] = None,
    n_disc_steps: Annotated[
        int | None, typer.Option("--n-disc-steps", "-k", min=1)
    ] = None,
    lr_g: Annotated[float | None, typer.Option("--lr-g", min=0.0)] = None,
    lr_d: Annotated[float | None, typer.Option("--lr-d", min=0.0)] = None,
    lambda_dispersion: Annotated[
        float | None, typer.Option("--lambda-dispersion", min=0.0)
    ] = None,
    data_seed: Annotated[int | None, typer.Option()] = None,
    init_seed: Annotated[int | None, typer.Option()] = None,
    resample: Annotated[Resample | None, typer.Option()] = None,
) -> None:
    """Train one (bootstrap dataset, init seed) cell of the design.

    Settings come from the design's frozen `design.json`, not from the config
    layers: an edited `deconvolve.toml` must not be able to change what cell 30 of a
    64-cell array measures. An explicit flag still wins.
    """
    # Every option below is read back out of `settings`, not by name: that is
    # what lets an explicit flag override one frozen value without the other
    # seventeen falling back to their bare code defaults. They still have to be
    # declared as parameters for Typer to parse and validate them. Adding an
    # nineteenth option here also means adding it to `uncertainty_freeze_command`
    # --- and removing one from this tuple without also removing it there is the
    # silent way to break that pairing, not adding one.
    _ = (
        n_datasets,
        n_seeds,
        n_eval,
        dataset,
        variable,
        config,
        batch_size,
        n_samples,
        hidden_units,
        n_layers,
        n_epochs,
        n_disc_steps,
        lr_g,
        lr_d,
        lambda_dispersion,
        data_seed,
        init_seed,
        resample,
    )

    try:
        frozen: dict[str, Any] = load_frozen(design_dir)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(object=error)) from error
    _require_complete(frozen, design_dir)

    settings: dict[str, Any] = _resolve_cell_settings(ctx, frozen)
    config_setting: Any = settings["config"]

    _ = run_cell(
        cell,
        design_dir,
        DesignSpec(
            n_datasets=settings["n_datasets"],
            n_seeds=settings["n_seeds"],
            data_seed=settings["data_seed"],
            init_seed=settings["init_seed"],
        ),
        dataset=DatasetName(value=settings["dataset"]),
        variables=_canonical_variables(settings["variable"]),
        config=Path(config_setting) if config_setting is not None else None,
        n_samples=settings["n_samples"],
        n_eval=settings["n_eval"],
        batch_size=settings["batch_size"],
        hidden_units=settings["hidden_units"],
        n_layers=settings["n_layers"],
        n_epochs=settings["n_epochs"],
        n_disc_steps=settings["n_disc_steps"],
        lr_g=settings["lr_g"],
        lr_d=settings["lr_d"],
        lambda_dispersion=settings["lambda_dispersion"],
        resample=Resample(settings["resample"]),
    )


@uncertainty_app.command(name="collect")
def uncertainty_collect_command(
    ctx: typer.Context,
    design_dir: Annotated[Path, typer.Argument(help="Design directory to collect.")],
    n_datasets: Annotated[int | None, typer.Option("--n-datasets", "-B", min=2)] = None,
    n_seeds: Annotated[int | None, typer.Option("--n-seeds", "-S", min=2)] = None,
    n_bins: Annotated[int, typer.Option(min=2)] = 20,
    data_seed: Annotated[int | None, typer.Option()] = None,
    init_seed: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Decompose a finished design and write its table, npz and figure.

    The grid's shape comes from `design.json`, not these flags: an ambient
    `[uncertainty.collect]` in `deconvolve.toml` re-deriving `n_datasets`/`n_seeds`
    from ordinary config would let `collect` disagree with the grid `freeze`
    actually fixed, silently charging seed variance to the dataset axis (or
    the reverse). A flag typed on this line still wins, exactly as it does
    for `uncertainty run`.
    """
    # Read back out of `settings`, not by name: see the matching note on
    # `uncertainty_run_command`.
    _ = (n_datasets, n_seeds, data_seed, init_seed)

    try:
        frozen: dict[str, Any] = load_frozen(design_dir)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(object=error)) from error
    _require_complete(frozen, design_dir)

    settings: dict[str, Any] = _resolve_cell_settings(ctx, frozen)

    _ = collect(
        design_dir,
        DesignSpec(
            settings["n_datasets"],
            settings["n_seeds"],
            settings["data_seed"],
            settings["init_seed"],
        ),
        n_bins=n_bins,
    )


@app.command(name="leakage-check")
def leakage_check_command(
    poison: Annotated[bool, typer.Option("--poison/--clean", "-X/")] = False,
    sentinel: Annotated[float, typer.Option("--sentinel", "-S")] = POISON_SENTINEL,
    seed: int = 42,
    init_seed: int = 0,
) -> None:
    run_leakage_check(poison, sentinel, seed, init_seed)


@config_app.command(name="show")
def config_show_command(
    ctx: typer.Context,
    command: Annotated[
        str | None, typer.Argument(help="Limit the listing to one command.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show what the config files said, and where each value came from."""

    # The root callback tolerates a `ConfigError` only for this command, and
    # stashes whichever of the two this invocation produced -- never both --
    # on `ctx.meta`, shared by every context in this invocation's chain.
    error: ConfigError | None = ctx.meta.get("deconvolve.config_error")
    if error is not None:
        raise typer.BadParameter(str(object=error)) from error
    resolved: Resolved = ctx.meta["deconvolve.resolved"]
    render(resolved, os.environ, command=command, as_json=as_json)
