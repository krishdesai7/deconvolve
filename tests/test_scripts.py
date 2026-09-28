"""The SLURM scripts, checked for the two things that only fail on the cluster.

Nothing here submits anything. What it guards is a class of breakage the rest
of the suite cannot see at all: a shell script is never imported, never linted
by ruff and never type-checked, so a syntax error or a missing shell function
reaches a compute node intact and costs an allocation to discover.

Both checks exist because both have already happened:

* `module load cudatoolkit/12.9` died with `command not found: module`. It is a
  shell function a site defines in an rc file, and a batch script runs zsh
  non-interactive and non-login, which does not read those -- so the function is
  present at a prompt and absent in the job. `scripts/_lmod.zsh` initialises it;
  the test is that every script calling `module` actually sources that first.
* `{ ... } always { ... }` does not run under `set -e`, so a cleanup written
  that way silently does not happen.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

SCRIPTS_DIR: Path = Path(__file__).resolve().parent.parent / "scripts"

# `_lmod.zsh` is sourced, never run, so it has no shebang and calls `exit` at
# top level; it is checked for syntax but excluded from the rest.
LMOD_HELPER: str = "_lmod.zsh"

# The hard-coded prefixes the helper falls back to when `MODULESHOME` is unset.
_LMOD_FALLBACKS: tuple[str, ...] = (
    "/usr/share/lmod/lmod/init/zsh",
    "/opt/cray/pe/lmod/lmod/init/zsh",
)


def _scripts() -> Sequence[Path]:
    return sorted(SCRIPTS_DIR.glob("*.zsh"))


def test_there_are_scripts_to_check() -> None:
    """A glob that silently matches nothing would pass every test below."""
    assert _scripts()


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_the_script_parses(script: Path) -> None:
    """`zsh -n` on every script.

    A shell script is not covered by ruff, pyrefly or an import, so a typo in
    one survives every gate this project has and fails on a compute node.
    """
    zsh: str | None = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is not installed")

    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [zsh, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script",
    [p for p in _scripts() if p.name != LMOD_HELPER],
    ids=lambda p: p.name,
)
def test_a_script_using_module_initialises_it_first(script: Path) -> None:
    r"""`module` must be defined before it is called.

    It is a shell function from Lmod, defined in a startup file that a
    non-interactive, non-login shell does not read -- which is what a SLURM
    batch script is. Without `scripts/_lmod.zsh` the first `module load` dies
    with `command not found: module`, and under `set -e` that takes the job with
    it, after the expensive part has already run.
    """
    text: str = script.read_text()
    uses: list[int] = [
        n
        for n, line in enumerate(text.splitlines(), start=1)
        # A call at the start of a line, not the word inside a comment.
        if re.match(r"\s*module\s+(load|unload|swap|purge)\b", line)
    ]
    if not uses:
        pytest.skip("does not use `module`")

    sourced: int | None = next(
        (
            n
            for n, line in enumerate(text.splitlines(), start=1)
            if re.search(rf"source\s+.*{re.escape(LMOD_HELPER)}", line)
        ),
        None,
    )
    assert sourced is not None, (
        f"{script.name} calls `module` at line {uses[0]} but never sources "
        f"scripts/{LMOD_HELPER}; it will die with `command not found: module`."
    )
    assert sourced < uses[0], (
        f"{script.name} sources scripts/{LMOD_HELPER} at line {sourced}, after "
        f"its first `module` call at line {uses[0]}."
    )


class TestLmodHelper:
    """The helper itself, whose whole job is to be defensive."""

    def test_it_tries_more_than_one_location(self) -> None:
        """`MODULESHOME` first, then the generic and Cray prefixes.

        Perlmutter is a Cray system and does not keep Lmod where the generic
        prefix says, so a single hard-coded fallback would be no fallback.
        """
        text: str = (SCRIPTS_DIR / LMOD_HELPER).read_text()
        assert "MODULESHOME" in text
        assert "/usr/share/lmod/lmod/init/zsh" in text
        assert "/opt/cray/pe/lmod/lmod/init/zsh" in text

    def test_it_fails_loudly_when_nothing_works(self, tmp_path: Path) -> None:
        """Silence here would mean the module load fails much later instead.

        "Nothing works" has to be arranged, not assumed. Unsetting `MODULESHOME`
        alone is not enough on a machine that has Lmod: the helper falls back
        to the generic and Cray prefixes, finds a real init script there, and
        succeeds -- which is its job. On Perlmutter this test failed for exactly
        that reason. So the fallbacks are pointed somewhere that cannot exist,
        and zsh gets a bare environment: no inherited `MODULESHOME`, no `HOME`
        with a `.zshenv` that might define `module`, no `PATH` with a `module`
        binary on it.
        """
        zsh: str | None = shutil.which("zsh")
        if zsh is None:
            pytest.skip("zsh is not installed")

        missing: Path = tmp_path / "missing"
        helper: str = (SCRIPTS_DIR / LMOD_HELPER).read_text()
        for fallback in _LMOD_FALLBACKS:
            assert fallback in helper, f"{LMOD_HELPER} no longer tries {fallback}"
            helper = helper.replace(fallback, f"{missing}{fallback}")
        isolated: Path = tmp_path / LMOD_HELPER
        _ = isolated.write_text(helper)

        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                zsh,
                "-c",
                (
                    "set -euo pipefail\n"
                    # `/etc/zshenv` is read even so, and a site may define
                    # `module` there.
                    "unfunction module 2>/dev/null || true\n"
                    "unalias module 2>/dev/null || true\n"
                    f"source {isolated}\n"
                    "print 'REACHED'"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env={"HOME": str(tmp_path), "PATH": str(missing)},
        )

        assert result.returncode != 0
        assert "REACHED" not in result.stdout
        assert "module system" in result.stderr

    def test_it_leaves_an_existing_module_alone(self) -> None:
        """A site that already defines `module` must not be re-sourced over."""
        zsh: str | None = shutil.which("zsh")
        if zsh is None:
            pytest.skip("zsh is not installed")

        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                zsh,
                "-c",
                (
                    "set -euo pipefail\n"
                    'module() { print "kept $*"; }\n'
                    f"source {SCRIPTS_DIR / LMOD_HELPER}\n"
                    "module load texlive"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        assert result.returncode == 0, result.stderr
        assert "kept load texlive" in result.stdout
