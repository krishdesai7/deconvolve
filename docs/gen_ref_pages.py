"""Generate the API reference pages from the source tree at build time.

Run by the `gen-files` plugin on every `mkdocs build`/`serve`; nothing it
writes lands on disk. Every public module under `src/deconvolve/` gets a page
holding a single mkdocstrings directive, every package gets an index page
listing its modules, and `SUMMARY.md` feeds the `literate-nav` plugin so the
nav tracks the tree too. Modules or packages whose name starts with `_` are
private and skipped.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent / "src"
PACKAGE = "deconvolve"
OUT = Path("api")


def is_public(parts: tuple[str, ...]) -> bool:
    return not any(part.startswith("_") for part in parts)


def title(parts: tuple[str, ...]) -> str:
    return parts[-1].replace("_", " ").title() if len(parts) > 1 else "Overview"


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


for pkg in packages:
    ident = ".".join(pkg)
    with mkdocs_gen_files.open(OUT / doc_path(pkg, package=True), "w") as fd:
        if len(pkg) == 1:
            fd.write(
                "# API Reference\n\n"
                "Generated from the source tree, type annotations and docstrings "
                "on every docs build.\n\n"
            )
        else:
            # `members: false`: packages re-export their submodules' names,
            # which are documented on the submodule pages.
            fd.write(f"::: {ident}\n    options:\n      members: false\n\n")
        fd.write("| Module | |\n| :--- | :--- |\n")
        for child, is_pkg in children(pkg):
            kind = "package" if is_pkg else "module"
            fd.write(f"| {link(child, package=is_pkg, base=pkg)} | {kind} |\n")
    mkdocs_gen_files.set_edit_path(OUT / doc_path(pkg, package=True), ROOT.joinpath(*pkg, "__init__.py"))

for mod in modules:
    with mkdocs_gen_files.open(OUT / doc_path(mod, package=False), "w") as fd:
        fd.write(f"::: {'.'.join(mod)}\n")
    mkdocs_gen_files.set_edit_path(OUT / doc_path(mod, package=False), ROOT.joinpath(*mod).with_suffix(".py"))


def write_nav(fd, pkg: tuple[str, ...], depth: int) -> None:
    indent = "    " * depth
    for child, is_pkg in children(pkg):
        target = doc_path(child, package=is_pkg).as_posix()
        fd.write(f"{indent}* [{title(child)}]({target})\n")
        if is_pkg:
            write_nav(fd, child, depth + 1)


with mkdocs_gen_files.open(OUT / "SUMMARY.md", "w") as fd:
    fd.write(f"* [{title((PACKAGE,))}](index.md)\n")
    write_nav(fd, (PACKAGE,), 0)
