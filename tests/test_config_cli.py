from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING, cast

import pytest
from deconvolve.cli import app
from deconvolve.config import Layer, Resolved, origins_for
from deconvolve.config.show import render
from rich.console import Console
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

    import typer
    from deconvolve.uncertainty import DesignSpec

runner: CliRunner = CliRunner()

# Read at import, not through the CLI; see "Deferred" in configuration.md.
_ENVIRONMENT_ONLY: frozenset[str] = frozenset(
    {"DECONVOLVE_CACHE_DIR", "DECONVOLVE_TIMING"}
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git-root-bounded directory that is the process's cwd, with no global layer."""
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-global"))
    # Layer 4 reads every `DECONVOLVE_<COMMAND>_<OPTION>`, so anything exported
    # in the shell running the suite would otherwise leak into these tests.
    for name in os.environ:
        if name.startswith("DECONVOLVE_") and name not in _ENVIRONMENT_ONLY:
            monkeypatch.delenv(name)
    return tmp_path


def test_help_reports_the_configured_default(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    # Not a bare "500": `--n-samples` always prints `[default: 500000]`, which
    # contains "500" and made this assertion pass with no config file at all.
    assert "[default: 500]" in result.stdout


def test_help_reports_the_code_default_without_config(project: Path) -> None:
    _ = project  # requested only to chdir into an isolated, config-free directory
    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    # Not a bare "100": several other options' defaults contain that substring.
    assert "[default: 100]" in result.stdout


def test_a_malformed_config_fails_the_command(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text("[train\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code != 0


def test_an_unknown_key_fails_the_command(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text("[train]\nn-epoch = 500\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code != 0


def test_a_pyproject_tool_ran_table_is_read(project: Path) -> None:
    _ = (project / "pyproject.toml").write_text(
        "[tool.deconvolve.train]\nn-epochs = 321\n"
    )

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "321" in result.stdout


def test_an_out_of_range_config_value_is_rejected_by_click(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text("[train]\nn-epochs = -5\n")

    result = runner.invoke(app, ["train"])

    assert result.exit_code != 0
    assert "n-epochs" in result.output


def test_config_show_lists_values_and_origins(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "train.n-epochs" in result.stdout
    assert "500" in result.stdout
    assert "deconvolve.toml" in result.stdout


def test_config_show_reports_the_environment_only_settings(project: Path) -> None:
    _ = project  # requested only to chdir into an isolated, config-free directory
    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "DECONVOLVE_CACHE_DIR" in result.stdout
    assert "DECONVOLVE_TIMING" in result.stdout


def test_config_show_explains_a_broken_config_instead_of_dying(project: Path) -> None:
    """The command that diagnoses a bad file must survive a bad file."""
    _ = (project / "deconvolve.toml").write_text("[train\n")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code != 0
    assert "invalid" in result.output
    assert "TOML" in result.output


def test_config_show_scopes_to_one_command(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text(
        "[train]\nn-epochs = 500\n\n[uncertainty.collect]\nn-bins = 40\n"
    )

    result = runner.invoke(app, ["config", "show", "train"])

    assert result.exit_code == 0
    assert "train.n-epochs" in result.stdout
    assert "n-bins" not in result.stdout


def test_config_show_disambiguates_two_layers_named_ran_toml(project: Path) -> None:
    """The global and project layers are both usually called `deconvolve.toml`.

    Rendering both as bare `deconvolve.toml` makes two different files look like
    the same origin -- exactly the bug this table exists to prevent.
    """
    global_dir = project / "xdg" / "deconvolve"
    global_dir.mkdir(parents=True)
    _ = (global_dir / "deconvolve.toml").write_text("[train]\nn-layers = 7\n")
    monkeypatch_env = project / "deconvolve.toml"
    _ = monkeypatch_env.write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(
        app,
        ["config", "show"],
        env={"XDG_CONFIG_HOME": str(project / "xdg")},
    )

    assert result.exit_code == 0
    assert "deconvolve.toml (global)" in result.stdout
    assert "deconvolve.toml (project)" in result.stdout


def test_config_show_json_is_machine_readable(project: Path) -> None:
    import json

    _ = (project / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["values"]["train"]["n-epochs"] == 500
    assert "deconvolve.toml" in payload["origins"]["train"]["n-epochs"]


def test_config_show_json_stays_plain_on_a_forced_terminal() -> None:
    """`FORCE_COLOR` (or CI) forces a terminal; `--json` must still parse."""
    buffer = io.StringIO()
    resolved = Resolved(
        values={("train",): {"n_epochs": 500}},
        origins={("train",): {"n_epochs": "deconvolve.toml:/p/deconvolve.toml"}},
        layers=(Layer(origin="deconvolve.toml:/p/deconvolve.toml", data={}),),
    )

    render(
        resolved,
        {},
        console=Console(file=buffer, force_terminal=True, color_system="truecolor"),
        as_json=True,
    )

    assert "\x1b[" not in buffer.getvalue()
    assert json.loads(buffer.getvalue())["values"]["train"]["n-epochs"] == 500


def test_config_show_as_json_is_not_layerable(project: Path) -> None:
    """M-7: none of `config show`'s own options may come from a config file.

    Not `--help`: the `config` group tolerates a `ConfigError` so `config
    show` can diagnose it, and `--help` short-circuits before that diagnosis
    runs, which would make this pass for the wrong reason.
    """
    _ = (project / "deconvolve.toml").write_text("[config.show]\nas-json = true\n")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code != 0
    assert "as-json" in result.output


def test_config_show_json_honours_the_command_scope(project: Path) -> None:
    """M-9: `--json` must scope like the table does, not emit every command."""
    import json

    _ = (project / "deconvolve.toml").write_text(
        "[train]\nn-epochs = 500\n\n[uncertainty.collect]\nn-bins = 40\n"
    )

    result = runner.invoke(app, ["config", "show", "train", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "train" in payload["values"]
    assert "uncertainty.collect" not in payload["values"]
    assert "uncertainty.collect" not in payload["origins"]


def test_origins_for_prefers_the_click_source_over_the_file() -> None:
    """A flag beats a file, and the recorded origin has to say so."""

    class _Ctx:
        def __init__(self) -> None:
            self._sources = {
                "n_epochs": "COMMANDLINE",
                "lr_g": "DEFAULT_MAP",
                "n_layers": "DEFAULT",
            }

        def get_parameter_source(self, name: str) -> object:
            return type("S", (), {"name": self._sources[name]})()

    resolved = Resolved(
        values={("train",): {"lr_g": 0.1}},
        origins={("train",): {"lr_g": "deconvolve.toml:/p/deconvolve.toml"}},
        layers=(Layer(origin="deconvolve.toml:/p/deconvolve.toml", data={}),),
    )

    origins = origins_for(
        _Ctx(), resolved, ("train",), names=("n_epochs", "lr_g", "n_layers")
    )

    assert origins == {
        "n_epochs": "command-line",
        "lr_g": "deconvolve.toml:/p/deconvolve.toml",
        "n_layers": "default",
    }


def test_uncertainty_run_without_a_frozen_design_names_freeze(project: Path) -> None:
    design = project / "design"
    design.mkdir()

    result = runner.invoke(app, ["uncertainty", "run", "0", str(design)])

    assert result.exit_code != 0
    assert "freeze" in result.output


def test_freeze_writes_the_resolved_values(project: Path) -> None:
    _ = (project / "deconvolve.toml").write_text("[uncertainty.freeze]\nn-epochs = 7\n")
    design = project / "design"

    result = runner.invoke(app, ["uncertainty", "freeze", str(design)])

    assert result.exit_code == 0
    import json

    frozen = json.loads((design / "design.json").read_text())
    assert frozen["config"]["n_epochs"] == 7
    assert "deconvolve.toml" in frozen["_origin"]["n_epochs"]


def test_freeze_refuses_to_overwrite_without_force(project: Path) -> None:
    design = project / "design"
    first = runner.invoke(app, ["uncertainty", "freeze", str(design)])
    assert first.exit_code == 0

    result = runner.invoke(app, ["uncertainty", "freeze", str(design)])

    assert result.exit_code != 0
    assert "--force" in result.output


def test_freeze_overwrites_with_force(project: Path) -> None:
    design = project / "design"
    first = runner.invoke(app, ["uncertainty", "freeze", str(design)])
    assert first.exit_code == 0

    result = runner.invoke(app, ["uncertainty", "freeze", str(design), "--force"])

    assert result.exit_code == 0


def test_a_config_edit_after_freeze_does_not_reach_a_cell(project: Path) -> None:
    """The regression test for the property the freeze path exists to protect."""
    import json

    _ = (project / "deconvolve.toml").write_text("[uncertainty.freeze]\nn-epochs = 7\n")
    design = project / "design"
    first = runner.invoke(app, ["uncertainty", "freeze", str(design)])
    assert first.exit_code == 0

    _ = (project / "deconvolve.toml").write_text(
        "[uncertainty.freeze]\nn-epochs = 999\n"
    )

    frozen = json.loads((design / "design.json").read_text())
    assert frozen["config"]["n_epochs"] == 7


def test_a_flag_overrides_a_frozen_value(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing that still beats the frozen file: a flag on this line.

    `cli.py` binds `run_cell` at import (`from .uncertainty import ...
    run_cell`), so it has to be patched as a name in `deconvolve.cli`; patching
    the `deconvolve.uncertainty` module would leave the bound name untouched.
    """
    from deconvolve import cli

    seen: list[object] = []

    def fake_run_cell(
        cell: int, design_dir: Path, spec: object, /, **kwargs: object
    ) -> Path:
        _ = (cell, spec)
        seen.append(kwargs["n_epochs"])
        return design_dir / "cell_0000.npz"

    monkeypatch.setattr(cli, "run_cell", fake_run_cell)

    design = project / "design"
    freeze = runner.invoke(
        app, ["uncertainty", "freeze", str(design), "--n-epochs", "77"]
    )
    assert freeze.exit_code == 0

    bare = runner.invoke(app, ["uncertainty", "run", "0", str(design)])
    assert bare.exit_code == 0, bare.output
    assert seen[-1] == 77

    flagged = runner.invoke(
        app,
        ["uncertainty", "run", "0", str(design), "--n-epochs", "3"],
    )
    assert flagged.exit_code == 0, flagged.output
    assert seen[-1] == 3


def test_environment_does_not_override_a_frozen_value(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported variable must never split a design the way a flag legitimately can.

    `_gate_autoenv` gives `uncertainty run` no variables at all, so
    `DECONVOLVE_UNCERTAINTY_RUN_N_EPOCHS` is never read; and even if it were,
    `_resolve_cell_settings` only lets a COMMANDLINE source beat the frozen
    file. Both halves are pinned: end to end here, and directly below.
    """
    from deconvolve import cli

    seen: list[object] = []

    def fake_run_cell(
        cell: int, design_dir: Path, spec: object, /, **kwargs: object
    ) -> Path:
        _ = (cell, spec)
        seen.append(kwargs["n_epochs"])
        return design_dir / "cell_0000.npz"

    monkeypatch.setattr(cli, "run_cell", fake_run_cell)

    design = project / "design"
    freeze = runner.invoke(
        app, ["uncertainty", "freeze", str(design), "--n-epochs", "77"]
    )
    assert freeze.exit_code == 0

    result = runner.invoke(
        app,
        ["uncertainty", "run", "0", str(design)],
        env={"DECONVOLVE_UNCERTAINTY_RUN_N_EPOCHS": "3"},
    )

    assert result.exit_code == 0, result.output
    assert seen[-1] == 77


def test_an_environment_source_does_not_override_a_frozen_value() -> None:
    """The second line of defence, should a variable ever reach `run` again."""
    from deconvolve.cli import _resolve_cell_settings

    class _Ctx:
        def __init__(self) -> None:
            self.params: dict[str, object] = {"n_epochs": 3}

        def get_parameter_source(self, name: str) -> object:
            sources = {"n_epochs": "ENVIRONMENT"}
            return type("S", (), {"name": sources[name]})()

    settings = _resolve_cell_settings(cast("typer.Context", _Ctx()), {"n_epochs": 77})

    assert settings["n_epochs"] == 77


def test_the_environment_overrides_a_config_file(project: Path) -> None:
    import json

    _ = (project / "deconvolve.toml").write_text("[uncertainty.freeze]\nn-epochs = 7\n")
    design = project / "design"

    result = runner.invoke(
        app,
        ["uncertainty", "freeze", str(design)],
        env={"DECONVOLVE_UNCERTAINTY_FREEZE_N_EPOCHS": "11"},
    )

    assert result.exit_code == 0, result.output
    frozen = json.loads((design / "design.json").read_text())
    assert frozen["config"]["n_epochs"] == 11
    assert frozen["_origin"]["n_epochs"] == "environment"


def test_a_flag_overrides_the_environment(project: Path) -> None:
    import json

    design = project / "design"

    result = runner.invoke(
        app,
        ["uncertainty", "freeze", str(design), "--n-epochs", "13"],
        env={"DECONVOLVE_UNCERTAINTY_FREEZE_N_EPOCHS": "11"},
    )

    assert result.exit_code == 0, result.output
    frozen = json.loads((design / "design.json").read_text())
    assert frozen["config"]["n_epochs"] == 13
    assert frozen["_origin"]["n_epochs"] == "command-line"


def test_a_not_layerable_option_has_no_environment_variable(project: Path) -> None:
    """`--force` is typed each time, not inherited from a shell profile."""
    design = project / "design"
    first = runner.invoke(app, ["uncertainty", "freeze", str(design)])
    assert first.exit_code == 0

    result = runner.invoke(
        app,
        ["uncertainty", "freeze", str(design)],
        env={"DECONVOLVE_UNCERTAINTY_FREEZE_FORCE": "1"},
    )

    assert result.exit_code != 0
    assert "--force" in result.output


@pytest.mark.usefixtures("project")
def test_help_advertises_exactly_the_layerable_variables() -> None:
    root = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    train = runner.invoke(app, ["train", "--help"], env={"COLUMNS": "200"})
    cell = runner.invoke(app, ["uncertainty", "run", "--help"], env={"COLUMNS": "200"})

    assert "DECONVOLVE_LOG_LEVEL" in root.stdout
    # Typer's own completion flags are in no spec: exported, they would fire
    # on every invocation.
    assert "DECONVOLVE_INSTALL_COMPLETION" not in root.stdout
    assert "DECONVOLVE_TRAIN_N_EPOCHS" in train.stdout
    assert "DECONVOLVE_TRAIN_LOAD_RUN" not in train.stdout
    assert "DECONVOLVE_" not in cell.stdout


def test_freeze_and_run_take_the_same_options() -> None:
    """The pairing `uncertainty_run_command` assumes: an option added to one
    command belongs on both.

    A contributor who adds `--foo` to `run` and reads it as `settings["foo"]`
    gets a loud `KeyError` if `freeze` does not also grow it. One who instead
    wires `foo=foo` straight through bypasses the frozen file silently --- this
    guard is against that second, quieter path.
    """
    from inspect import signature

    from deconvolve.cli import uncertainty_freeze_command, uncertainty_run_command

    freeze_params = set(signature(uncertainty_freeze_command).parameters) - {
        "ctx",
        "force",
    }
    run_params = set(signature(uncertainty_run_command).parameters) - {"ctx", "cell"}

    assert freeze_params == run_params


def test_freeze_writes_exactly_the_spec_declares(project: Path) -> None:
    """`freeze`'s literal `values` dict must name the same keys the spec does.

    A key present in one but not the other falls silently out of the frozen
    file, or out of what `origins_for` records for it, without either side
    raising anything.
    """
    import json
    from typing import Any

    from deconvolve.config.spec import build_spec

    design = project / "design"
    result = runner.invoke(app, ["uncertainty", "freeze", str(design)])
    assert result.exit_code == 0

    frozen: dict[str, Any] = json.loads((design / "design.json").read_text())
    expected = set(build_spec(app).children["uncertainty"].children["freeze"].options)

    assert set(frozen["config"]) == expected


def test_freeze_force_refuses_once_cells_exist(project: Path) -> None:
    """`--force` may only rewrite settings nothing has trained under yet."""
    design = project / "design"
    first = runner.invoke(app, ["uncertainty", "freeze", str(design)])
    assert first.exit_code == 0
    _ = (design / "cell_0000.npz").write_bytes(b"")

    result = runner.invoke(app, ["uncertainty", "freeze", str(design), "--force"])

    assert result.exit_code != 0
    assert "cell" in result.output.lower()


def test_run_with_a_truncated_design_names_the_missing_keys(project: Path) -> None:
    """A partial `design.json` is diagnosed by name, not a bare `KeyError`."""
    import json

    design = project / "design"
    design.mkdir()
    _ = (design / "design.json").write_text(
        json.dumps({"config": {"n_epochs": 7}, "_origin": {}})
    )

    result = runner.invoke(app, ["uncertainty", "run", "0", str(design)])

    assert result.exit_code != 0
    assert "n_datasets" in result.output
    assert "freeze" in result.output


def test_uncertainty_collect_without_a_frozen_design_names_freeze(
    project: Path,
) -> None:
    """`collect` must fail exactly as loudly as `run` does when unfrozen."""
    design = project / "design"
    design.mkdir()

    result = runner.invoke(app, ["uncertainty", "collect", str(design)])

    assert result.exit_code != 0
    assert "freeze" in result.output


def test_uncertainty_collect_uses_the_frozen_grid_shape_not_the_flags(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-1's regression test: an ambient/bare-flag shape must lose to `design.json`.

    Before this fix, `collect` built its `DesignSpec` straight from
    `n_datasets`/`n_seeds` layered off the ordinary config stack, so a design
    frozen at one grid shape but `collect`-ed under another silently
    misattributed variance between the bootstrap and seed axes.
    """
    from deconvolve import cli

    design = project / "design"
    freeze = runner.invoke(
        app, ["uncertainty", "freeze", str(design), "-B", "2", "-S", "6"]
    )
    assert freeze.exit_code == 0, freeze.output

    seen: list[DesignSpec] = []

    def fake_collect(
        design_dir: Path, spec: DesignSpec, /, **kwargs: object
    ) -> dict[str, object]:
        _ = (design_dir, kwargs)
        seen.append(spec)
        return {}

    monkeypatch.setattr(cli, "collect", fake_collect)

    # The code default for `-B`/`-S` (8x8) differs from the frozen 2x6 grid.
    # If `collect` were still deriving its shape from these flags/config
    # rather than `design.json`, this would decompose the wrong grid.
    result = runner.invoke(app, ["uncertainty", "collect", str(design)])

    assert result.exit_code == 0, result.output
    spec = seen[-1]
    assert (spec.n_datasets, spec.n_seeds) == (2, 6)


def test_ambient_config_cannot_reach_the_collect_grid_shape(project: Path) -> None:
    """C-1's other half: no config file may populate these names either.

    Before the fix, `n_datasets`/`n_seeds`/`data_seed`/`init_seed` on
    `uncertainty collect` were ordinary layerable options, so `[uncertainty.
    collect]` in `deconvolve.toml` could silently re-derive a design's grid shape
    even without a stale flag anywhere in sight.
    """
    _ = (project / "deconvolve.toml").write_text("[uncertainty.collect]\nn-seeds = 4\n")

    result = runner.invoke(app, ["uncertainty", "collect", "--help"])

    assert result.exit_code != 0
    assert "n-seeds" in result.output
