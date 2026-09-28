"""Generate the API reference pages under `docs/api/` from the source tree.

Zensical has no build hook for generated pages, so the `doc-build` and
`doc-serve` recipes run this first. `docs/api/` is rebuilt from scratch every
time and is not tracked. Every public module under `src/deconvolve/` gets a
page holding a single mkdocstrings directive, every package gets an index page
listing its modules, and `SUMMARY.md` feeds the `literate-nav` plugin so the
nav tracks the tree too. Modules or packages whose name starts with `_` are
private and skipped.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "src"
OUT = REPO / "docs" / "api"
PACKAGE = "deconvolve"


def is_public(parts: tuple[str, ...]) -> bool:
    return not any(part.startswith("_") for part in parts)


def title(parts: tuple[str, ...]) -> str:
    return parts[-1] if len(parts) > 1 else "Overview"


modules: list[tuple[str, ...]] = []
packages: list[tuple[str, ...]] = []
for path in sorted((ROOT / PACKAGE).rglob("*.py")):
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        if is_public(parts[:-1]):
            packages.append(parts[:-1])
    elif is_public(parts):
        modules.append(parts)


def doc_path(parts: tuple[str, ...], *, package: bool) -> Path:
    # Drop the leading `deconvolve/` so URLs read `api/data/datasets/`.
    rel = Path(*parts[1:])
    return rel / "index.md" if package else rel.with_suffix(".md")


def children(pkg: tuple[str, ...]) -> list[tuple[tuple[str, ...], bool]]:
    subpackages = [(p, True) for p in packages if p[:-1] == pkg]
    submodules = [(m, False) for m in modules if m[:-1] == pkg]
    return sorted(subpackages + submodules, key=lambda item: (not item[1], item[0]))


def link(parts: tuple[str, ...], *, package: bool, base: tuple[str, ...]) -> str:
    target = doc_path(parts, package=package).as_posix()
    here = doc_path(base, package=True).parent.as_posix()
    rel = target if here == "." else Path(target).relative_to(here).as_posix()
    return f"[`{'.'.join(parts)}`]({rel})"


def write(rel: Path, text: str) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text)


def package_page(pkg: tuple[str, ...]) -> str:
    if len(pkg) == 1:
        head = (
            "# API Reference\n\n"
            f"::: {PACKAGE}\n"
            "    options:\n"
            "      members: false\n"
            "      show_root_heading: false\n\n"
        )
    else:
        # `members: false`: packages re-export their submodules' names,
        # which are documented on the submodule pages.
        head = f"::: {'.'.join(pkg)}\n    options:\n      members: false\n\n"
    rows = "".join(
        f"| {link(child, package=is_pkg, base=pkg)} | "
        f"{'package' if is_pkg else 'module'} |\n"
        for child, is_pkg in children(pkg)
    )
    return f"{head}| Module | |\n| :--- | :--- |\n{rows}"


def nav(pkg: tuple[str, ...], depth: int) -> str:
    lines = []
    for child, is_pkg in children(pkg):
        target = doc_path(child, package=is_pkg).as_posix()
        lines.append(f"{'    ' * depth}* [{title(child)}]({target})\n")
        if is_pkg:
            lines.append(nav(child, depth + 1))
    return "".join(lines)


shutil.rmtree(OUT, ignore_errors=True)
for pkg in packages:
    write(doc_path(pkg, package=True), package_page(pkg))
for mod in modules:
    write(doc_path(mod, package=False), f"::: {'.'.join(mod)}\n")
write(Path("SUMMARY.md"), f"* [{title((PACKAGE,))}](index.md)\n{nav((PACKAGE,), 0)}")
