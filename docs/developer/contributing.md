# Contributing & Tooling

Deconvolve enforces rigorous code quality, formatting, typing, and testing standards.

---

## Development Workflow with `just`

The repository includes a [`Justfile`](https://github.com/casey/just) defining all developer commands:

```shell
# List all available recipes
just
```

### Validation & Checks

Before submitting changes, run the validation suite:

```shell
# Run all local validation (formatting, linting, type-checking, complexity, tests)
just validate

# Run full CI suite (validation + dependency security audit)
just ci
```

Individual checks:

- **Format check**: `just format` (`uv format --check`)
- **Lint check**: `just lint` (`uv run --locked ruff check`)
- **Type check**: `just typecheck` (`uv run --locked pyrefly check`)
- **Complexity check**: `just complexity` (`uv run --locked complexipy --suggest-refactors`)
- **Unit tests**: `just test` or `just test-fast`

### Automated Fixes

```shell
# Apply safe lint fixes and reformat code
just lint-fix

# Apply unsafe lint fixes and reformat code
just lint-fix-unsafe

# Infer type annotations and imports automatically
just infer
```

---

## Documentation Commands

Build and preview the documentation locally:

```shell
# Start local live-reload documentation server
just docs-serve

# Build static documentation site to site/
just docs-build
```

---

## Code Quality Standards

- **Formatting**: `ruff format` with a line-length limit of 88.
- **Docstrings**: Formatted with `docstring-code-format = true`.
- **Cyclomatic Complexity**: Functions must maintain cyclomatic complexity $\le 10$, enforced by `complexipy`.
- **Strict Typing**: Type checking enforced by `pyrefly` with `treat-all-caps-as-final = true`.
- **Pre-commit**: Managed with `prek` (`.prek.toml`).
