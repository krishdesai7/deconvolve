from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deconvolve.cli import app
from deconvolve.config import ConfigError, Layer, default_map, discover, load
from deconvolve.config.spec import CommandSpec, build_spec

if TYPE_CHECKING:
    from pathlib import Path


def _repo(tmp_path: Path) -> Path:
    """A directory that looks like a git repository root."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_no_config_anywhere_discovers_nothing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")}) == ()


def test_project_ran_toml_is_found_from_a_subdirectory(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _ = (root / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")
    deep = root / "a" / "b"
    deep.mkdir(parents=True)

    layers = discover(cwd=deep, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}
    assert layers[0].origin == f"deconvolve.toml:{root / 'deconvolve.toml'}"


def test_the_walk_stops_at_the_git_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _ = (outside / "deconvolve.toml").write_text("[train]\nn-epochs = 1\n")
    root = outside / "repo"
    root.mkdir()
    (root / ".git").mkdir()

    assert discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")}) == ()


def test_a_git_file_bounds_the_walk_as_a_git_directory_does(tmp_path: Path) -> None:
    """A linked worktree's `.git` is a file, not a directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    _ = (outside / "deconvolve.toml").write_text("[train]\nn-epochs = 1\n")
    root = outside / "wt"
    root.mkdir()
    _ = (root / ".git").write_text("gitdir: /elsewhere\n")

    assert discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")}) == ()


def test_ran_toml_shadows_pyproject_in_the_same_directory(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _ = (root / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")
    _ = (root / "pyproject.toml").write_text(
        "[tool.deconvolve.train]\nn-epochs = 9\nlr-g = 0.1\n"
    )

    layers = discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}


def test_the_nearest_directory_wins_and_the_walk_stops(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _ = (root / "deconvolve.toml").write_text("[train]\nn-epochs = 1\n")
    near = root / "a"
    near.mkdir()
    _ = (near / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")

    layers = discover(cwd=near, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}


def test_a_pyproject_without_a_tool_ran_table_does_not_stop_the_walk(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    _ = (root / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")
    near = root / "a"
    near.mkdir()
    _ = (near / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')

    layers = discover(cwd=near, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}


def test_the_global_layer_comes_first(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    (xdg / "deconvolve").mkdir(parents=True)
    _ = (xdg / "deconvolve" / "deconvolve.toml").write_text("[train]\nlr-g = 0.1\n")
    root = _repo(tmp_path / "repo")
    _ = (root / "deconvolve.toml").write_text("[train]\nn-epochs = 500\n")

    layers = discover(cwd=root, environ={"XDG_CONFIG_HOME": str(xdg)})

    assert [layer.data for layer in layers] == [
        {"train": {"lr-g": 0.1}},
        {"train": {"n-epochs": 500}},
    ]


def test_xdg_config_home_defaults_to_dot_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".config" / "deconvolve").mkdir(parents=True)
    _ = (home / ".config" / "deconvolve" / "deconvolve.toml").write_text(
        "[train]\nlr-g = 0.1\n"
    )
    root = _repo(tmp_path / "repo")

    layers = discover(cwd=root, environ={"HOME": str(home)})

    assert [layer.data for layer in layers] == [{"train": {"lr-g": 0.1}}]


def test_malformed_toml_names_the_path_and_the_position(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _ = (root / "deconvolve.toml").write_text("[train\nn-epochs = 500\n")

    with pytest.raises(ConfigError) as excinfo:
        _ = discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    message = str(excinfo.value)
    assert str(root / "deconvolve.toml") in message
    assert "line 1" in message


def test_an_unreadable_file_names_the_path(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    broken = root / "deconvolve.toml"
    _ = broken.write_text("[train]\n")
    broken.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as excinfo:
            _ = discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})
        assert str(broken) in str(excinfo.value)
    finally:
        broken.chmod(0o644)


def test_the_spec_mirrors_the_command_tree() -> None:
    spec = build_spec(app)

    assert set(spec.children) == {
        "train",
        "evaluate",
        "plot",
        "report",
        "baseline",
        "uncertainty",
        "leakage-check",
        "config",
    }
    assert set(spec.children["baseline"].children) == {"ibu", "omnifold"}


def test_group_level_options_sit_at_the_root() -> None:
    assert "log_level" in build_spec(app).options


def test_hyperparameters_are_layerable() -> None:
    train = build_spec(app).children["train"]

    for name in ("n_epochs", "batch_size", "lr_g", "lr_d", "n_layers", "hidden_units"):
        assert name in train.options, name
    assert train.options["n_epochs"] is int


def test_positional_arguments_are_not_layerable() -> None:
    report = build_spec(app).children["report"]

    assert "run_dir" not in report.options
    assert "run_dir" in report.excluded


def test_force_and_identity_options_are_not_layerable() -> None:
    spec = build_spec(app)

    assert "force" in spec.children["evaluate"].excluded
    assert "load_run" in spec.children["train"].excluded


def test_uncertainty_run_is_excluded_wholesale() -> None:
    """Design cells resolve from design.json, never from an ambient file."""
    uncertainty = build_spec(app).children["uncertainty"]

    assert "run" not in uncertainty.children
    assert "collect" in uncertainty.children
    assert "freeze" in uncertainty.children


def _spec() -> CommandSpec:
    return CommandSpec(
        options={"log_level": str},
        children={
            "train": CommandSpec(
                options={"n_epochs": int, "lr_g": float, "n_layers": int},
                excluded=frozenset({"load_run"}),
            ),
            "baseline": CommandSpec(
                children={"omnifold": CommandSpec(options={"n_epochs": int})}
            ),
            "leakage-check": CommandSpec(options={"sentinel": float}),
        },
    )


def _layer(
    data: dict[str, object], origin: str = "deconvolve.toml:/p/deconvolve.toml"
) -> Layer:
    return Layer(origin=origin, data=data)


def test_keys_normalize_from_kebab_to_snake() -> None:
    resolved = load((_layer({"train": {"n-epochs": 500}}),), _spec())

    assert resolved.values["train",] == {"n_epochs": 500}


def test_underscores_are_accepted_too() -> None:
    resolved = load((_layer({"train": {"n_epochs": 500}}),), _spec())

    assert resolved.values["train",] == {"n_epochs": 500}


def test_nested_command_tables_resolve() -> None:
    resolved = load((_layer({"baseline": {"omnifold": {"n-epochs": 80}}}),), _spec())

    assert resolved.values["baseline", "omnifold"] == {"n_epochs": 80}


def test_group_level_options_resolve_at_the_root() -> None:
    resolved = load((_layer({"log-level": "debug"}),), _spec())

    assert resolved.values[()] == {"log_level": "debug"}


def test_a_later_layer_overrides_per_key_not_wholesale() -> None:
    resolved = load(
        (
            _layer({"train": {"lr-g": 0.1, "n-epochs": 1}}, "deconvolve.toml:/global"),
            _layer({"train": {"n-epochs": 500}}, "deconvolve.toml:/project"),
        ),
        _spec(),
    )

    assert resolved.values["train",] == {"lr_g": 0.1, "n_epochs": 500}
    assert resolved.origins["train",]["lr_g"] == "deconvolve.toml:/global"
    assert resolved.origins["train",]["n_epochs"] == "deconvolve.toml:/project"


def test_default_map_is_nested_by_command_path() -> None:
    resolved = load(
        (_layer({"log-level": "debug", "baseline": {"omnifold": {"n-epochs": 80}}}),),
        _spec(),
    )

    assert default_map(resolved) == {
        "log_level": "debug",
        "baseline": {"omnifold": {"n_epochs": 80}},
    }


def test_an_unknown_key_suggests_the_near_miss() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _ = load((_layer({"train": {"n-epoch": 500}}),), _spec())

    message = str(excinfo.value)
    assert "n-epoch" in message
    assert "n-epochs" in message
    assert "/p/deconvolve.toml" in message


def test_an_unknown_table_suggests_the_near_miss() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _ = load((_layer({"trian": {"n-epochs": 500}}),), _spec())

    assert "trian" in str(excinfo.value)
    assert "train" in str(excinfo.value)


def test_a_non_layerable_option_says_why() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _ = load((_layer({"train": {"load-run": "runs/x"}}),), _spec())

    message = str(excinfo.value)
    assert "load-run" in message
    assert "cannot be configured" in message


def test_a_non_integral_float_for_an_int_option_is_rejected() -> None:
    """Click would silently truncate 3.7 to 3."""
    with pytest.raises(ConfigError) as excinfo:
        _ = load((_layer({"train": {"n-layers": 3.7}}),), _spec())

    assert "n-layers" in str(excinfo.value)
    assert "3.7" in str(excinfo.value)


def test_an_integral_float_for_an_int_option_is_accepted() -> None:
    resolved = load((_layer({"train": {"n-layers": 3.0}}),), _spec())

    assert resolved.values["train",] == {"n_layers": 3}


def test_a_float_option_accepts_an_int() -> None:
    resolved = load((_layer({"train": {"lr-g": 1}}),), _spec())

    assert resolved.values["train",] == {"lr_g": 1}


def test_a_hyphenated_command_name_survives_the_merge() -> None:
    """The path key must be the registered command name, hyphen and all."""
    resolved = load((_layer({"leakage-check": {"sentinel": 9.5}}),), _spec())

    assert default_map(resolved) == {"leakage-check": {"sentinel": 9.5}}


def test_a_hyphenated_command_name_survives_into_the_default_map() -> None:
    """Click looks up default_map by the registered name, hyphen and all."""
    resolved = load((_layer({"leakage-check": {"sentinel": 9.5}}),), build_spec(app))

    assert default_map(resolved) == {"leakage-check": {"sentinel": 9.5}}
