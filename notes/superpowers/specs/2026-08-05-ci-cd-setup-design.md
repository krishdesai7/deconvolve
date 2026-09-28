# CI/CD Setup Design

## Goal

Give the repository automated, unbypassable enforcement of the checks that
already exist locally (`just check`, `just ci`), plus fast local feedback
before a commit is even made, plus a repeatable release process. Concretely:
a GitHub Actions CI workflow with inline PR annotations for lint/format/type/
complexity issues, a `pre-commit` configuration for local pre-flight checks,
and three release workflows built around `rooster` and PyPI Trusted
Publishing.

## Architecture

Two layers for CI, each covering a gap the other leaves open:

- **GitHub Actions** is the actual gate. It runs regardless of who pushes or
  from where, cannot be skipped with a flag, and is what branch protection
  points at.
- **pre-commit hooks** are a local convenience layer. They catch formatting,
  lint, and type errors before a commit is made, without waiting on a
  push/CI round trip. They are not a substitute for the Actions gate, because
  they can be bypassed (`--no-verify`, or simply not installed).

Plus a separate, independent piece: **release automation**. This is the only
thing resembling continuous delivery for a research library with no
deployment target — cutting a version bump, a changelog, a git tag, a GitHub
Release, and a PyPI upload, via `rooster` (already a `release` dependency
group in `pyproject.toml`) and PyPI Trusted Publishing.

For CI, dedicated per-tool GitHub Actions (`ruff-action`, pyrefly's action,
`complexipy-action`) are used instead of shelling out to the equivalent `just`
recipes, specifically for their inline PR annotations — each finding shows up
as a review comment on the exact line, not just a log line to scroll through.
This does mean each of those three tools' versions can, in principle, drift
from the version `uv.lock` resolves for local development (each action
installs its own copy rather than reusing the project's `uv sync`'d
environment). That trade-off is accepted deliberately here in exchange for
the annotation UX — see Alternatives Considered for the reasoning against it
that was overridden. `just check`/`just ci` remain unchanged and are still
the correct way to run everything locally; CI's use of dedicated actions for
three of the checks doesn't replace what those recipes do for a contributor
working locally.

## GitHub Actions Workflow

New file: `.github/workflows/ci.yml`.

- **Triggers:** `pull_request` (any branch) and `push` to `master`.
- **Job:** single job, `ubuntu-latest`, named `ci`. No matrix — one supported
  Python version (3.13, per `requires-python` in `pyproject.toml`) and one
  platform. Linux is chosen over macOS because it is what GitHub-hosted
  runners default to and is cheaper; a macOS job can be added later if
  platform-specific failures actually appear, rather than preemptively.
- **Steps:**
  1. `actions/checkout@v7`.
  2. `astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9` (`v9.0.0`),
     with caching enabled, to install `uv` itself. Both this and the
     `checkout` version are pinned to their current major versions (not
     older ones) because older releases of these actions run on Node
     versions GitHub Actions has deprecated, which surfaces as a runner
     warning on every run.
  3. `uv sync --locked` — installs the default dependency groups (`dev`:
     `just`, `pytest`, `complexipy`, etc.). The `release` group (`rooster`)
     is not needed here.
  4. `astral-sh/ruff-action@v4` (default args, i.e. `ruff check`) — lint,
     with inline PR annotations.
  5. `astral-sh/ruff-action@v4` with `args: "format --check --diff"` — format
     check, with inline PR annotations.
  6. `facebook/pyrefly@main` with `python-version: "3.13"` — type check, with
     inline PR annotations. (Pyrefly's action is only published at `@main`,
     not tagged releases — see Alternatives Considered.)
  7. `rohaquinlop/complexipy-action@v6` with `paths: '.'` — complexity check,
     with inline PR annotations.
  8. `just test`.
  9. `just audit`.

Steps 4–7 replace what `just lint` / `just format-check` / `just typecheck` /
`just complexity` do locally, using each tool's dedicated action instead so
that findings appear as inline PR review comments. Steps 8–9 are the two
remaining `just ci` recipes for which no equivalent action exists (or would
add value) and are run exactly as they would be locally. `just check`/
`just ci` are unchanged in the `Justfile` and remain the right way to run
everything in one shot locally — CI simply composes the same underlying
checks differently to get annotations for the four most-common-to-fail ones.

No separate `pre-commit run --all-files` job is added to this workflow — the
`ci` job above is the only required check. Pre-commit hooks are a local
convenience only; their absence in CI is intentional, not an oversight to fix
later.

## Pre-commit Configuration

New file: `.pre-commit-config.yaml`, covering the fast tier of checks:

- **`uv-lock`** (from `astral-sh/uv-pre-commit`) — keeps `uv.lock` in sync
  whenever `pyproject.toml` changes.
- **`ruff-check`** with `--fix`, and **`ruff-format`** (from
  `astral-sh/ruff-pre-commit`) — mirrors `just fix`. The hook's `rev` pins its
  own copy of `ruff`, independent of the `ruff` version `uv.lock` resolves;
  this must be bumped by hand alongside the `pyproject.toml` dependency to
  avoid the two drifting apart. There is no `language: system` option for
  this hook, so this small manual-sync cost is accepted rather than avoided.
- **`pyrefly-check`** (from `facebook/pyrefly-pre-commit`), configured with
  `language: system` and `pass_filenames: false` — runs the project's own
  `uv sync`'d `pyrefly` rather than a separately pinned copy, avoiding the
  drift risk above entirely.

Explicitly not included:

- **`ty` (`astral-sh/ty-pre-commit`)** — `ty` is a different type checker
  (Astral's) from the one this repo standardizes on (`pyrefly`, per the
  `Justfile` and `pyproject.toml`). Adding it would mean running two type
  checkers with potentially conflicting opinions, for no current benefit.
- **`complexipy`** (a hook does exist,
  [`rohaquinlop/complexipy-pre-commit`](https://github.com/rohaquinlop/complexipy-pre-commit))
  — still excluded from pre-commit even though its CI counterpart
  (`complexipy-action`) is now used. Complexity checks belong in the same
  "slower" tier as the test suite: appropriate for CI, not for a pre-commit
  hook that should stay fast. Using the action in CI but not the hook locally
  is a deliberate asymmetry, not an inconsistency to fix.
- **Generic hygiene hooks** (trailing whitespace, end-of-file-fixer, etc.
  from `pre-commit/pre-commit-hooks`) — not requested, and `ruff-format`
  already covers most of what would matter for `.py` files. Can be added
  later if noise shows up in practice.

Contributors install with `uv run pre-commit install` (or
`uvx pre-commit install`) once; hooks then run automatically on `git commit`.

## Branch Protection

On GitHub, under Settings → Branches, add a protection rule for `master`
requiring the `ci` status check to pass before merging. This is what turns
the Actions workflow from "informational" into an actual gate — without it,
a failing check is visible but does not block a merge.

## Release Automation

Three workflows, forming one pipeline from "decide to cut a release" to
"published on PyPI and GitHub." `rooster` (already configured in
`[tool.rooster]` in `pyproject.toml`) drives the version bump and changelog;
its `major-labels = []` / `minor-labels = ["breaking"]` configuration is
existing, deliberate policy for a pre-1.0 package (major is reserved for the
1.0 release itself) and is unchanged by this design.

### 1. `prepare-release.yml` — manually triggered

Dispatched by hand from the Actions tab (`workflow_dispatch`), with an
optional `bump` choice (`major`/`minor`/`patch`/`pre`; blank infers the bump
from merged PR labels since the last release, which is `rooster`'s normal
mode). Runs `uv sync --group release --frozen`, then `rooster release`
(`--bump "$BUMP"` if provided), which bumps the version in `pyproject.toml`
and updates `CHANGELOG.md` in the working tree. Reads the new version back
out of `pyproject.toml`, commits both changed files to a new `release/X.Y.Z`
branch, pushes it, and opens a PR against `master` via `gh pr create`. Needs
`contents: write` and `pull-requests: write` permissions; the default
`GITHUB_TOKEN` covers both.

This produces a normal, reviewable PR — the changelog and version bump are
visible and can be corrected before merge, rather than being pushed straight
to `master`.

### 2. `tag-release.yml` — new, triggers the actual release

Addresses the gap in the pasted `prepare-release.yml`/`release.yml` pair:
merging the release PR updates `master` but creates no tag, and `release.yml`
only triggers on a tag push. This workflow closes that gap automatically
(per your choice of auto-tagging over a manual `git tag` step):

- **Trigger:** `pull_request`, `types: [closed]`.
- **Condition:** `github.event.pull_request.merged == true` and
  `startsWith(github.event.pull_request.head.ref, 'release/')` — only fires
  for merged release PRs, not any other merged PR.
- **Steps:** check out `master`, read the version back out of `pyproject.toml`
  (now updated, since the release PR merged), then `git tag X.Y.Z` and
  `git push origin X.Y.Z`.
- **Permissions:** `contents: write`.

Pushing the tag is what triggers `release.yml` below.

### 3. `release.yml` — builds, publishes, and creates the GitHub Release

- **Trigger:** `push`, `tags: ["[0-9]+.[0-9]+.[0-9]+"]` — matching the bare
  `X.Y.Z` tag `tag-release.yml` pushes (no `v` prefix, consistent with how
  `version` is written in `pyproject.toml`).
- **Steps:** checkout, `setup-uv`, `uv sync --frozen`, `uv build` (produces
  sdist + wheel in `dist/`), `uv publish`, then `gh release create
  "$GITHUB_REF_NAME" dist/* --generate-notes`.
- **Permissions:** `contents: write` (for the GitHub Release) and
  `id-token: write` (for PyPI Trusted Publishing — see below).

**One-time manual prerequisite, done on pypi.org, not in this repository:**
`uv publish` authenticates via PyPI's Trusted Publishing (OIDC), which needs
a trusted publisher registered on the PyPI side before the first automated
publish will work. Since `ran` has no PyPI project yet, this uses PyPI's
"pending publisher" flow: register a pending trusted publisher for project
name `ran`, owner `krishdesai7`, repository `RANv4`, workflow filename
`release.yml`. PyPI creates the project automatically the first time that
workflow successfully publishes. No token or secret needs to be stored in
GitHub — that's the point of Trusted Publishing.

## Alternatives Considered

- **Running everything through `just ci` in one step**, instead of
  `ruff-action` / pyrefly's action / `complexipy-action`. This was the
  original recommendation: it guarantees CI checks the exact tool versions
  `uv.lock` resolves, with zero drift risk, at the cost of no inline PR
  annotations (just a log to scroll through). Superseded by the annotation
  UX being worth more here than the drift risk, per your call — noting for
  the record that `ruff-action` in particular defaults to reading its
  version from `pyproject.toml`, so of the three, it actually carries the
  least drift risk in practice; pyrefly's action and `complexipy-action`
  default to "latest" unless a `version` input is set, which is the real
  source of drift exposure if it ever becomes a problem worth revisiting.
- **`extractions/setup-just`**, instead of relying on `just` already being a
  `dev` dependency. Unnecessary: `uv sync` already installs `just` into the
  project's own environment, so `uv run just ci` needs no separate
  installation step.
- **Git pre-push hook running `just ci` locally**, instead of GitHub Actions.
  Rejected as the primary gate because it isn't versioned/shared without
  extra tooling, gives no visibility on GitHub (no check on PRs, no history),
  and — like pre-commit hooks — can be bypassed.
- **A third-party CI service** (CircleCI, Buildkite, etc.). No reason to
  leave GitHub Actions: the repository is already hosted on GitHub and the
  workload is a single fast job.
- **Running `pre-commit run --all-files` as a second required CI job.**
  Considered as a way to enforce hooks server-side against contributors who
  bypass them locally, but rejected for now: it would duplicate most of what
  `just ci`'s lint/format/typecheck steps already check, via a slightly
  different mechanism (pre-commit's own tool versions vs. `uv.lock`'s). Can
  be revisited if bypassing local hooks becomes an actual problem.

## Non-Goals

- A macOS (or any matrix) CI job. Revisit if platform-specific failures
  actually occur.
- Dependabot/Renovate or any automated dependency-update workflow.
- Publishing to any index other than PyPI (e.g. no TestPyPI dry-run step).
- SHA-pinning every third-party action for supply-chain hardening.
  `actions/checkout` and `astral-sh/setup-uv` are pinned by necessity (to
  dodge the Node-version deprecation warning); the others are referenced by
  tag/branch per their own docs. Revisit if that becomes a real concern.
