# List available recipes.
default:
    @printf '\033[1mAvailable recipes:\033[0m\n'
    @printf '  \033[1m\033[36mImmutable / read-only:\033[0m\n'
    @printf '  %-26s %s\n' '    ci'              '# Run the full CI validation suite.'
    @printf '  %-26s %s\n' '      validate'     '# Run all local, read-only validation.'
    @printf '  %-26s %s\n' '        format *args' '# Check formatting without modifying files.'
    @printf '  %-26s %s\n' '        lint'        '# Run Ruff lint checks.'
    @printf '  %-26s %s\n' '        typecheck'  '# Run Pyrefly type checks.'
    @printf '  %-26s %s\n' '        complexity'   '# Run complexity checks.'
    @printf '  %-26s %s\n' '        test *args'  '# Run tests, optionally forwarding arguments to pytest.'
    @printf '  %-26s %s\n' '      audit'        '# Audit locked dependencies for known vulnerabilities.'
    @printf '  %-26s %s\n' '    test-fast *args' '# Run tests, minus the slow ones.'
    @printf '  \033[1m\033[33mMutable / writes:\033[0m\n'
    @printf '  %-26s %s\n' '    lint-fix'        '# Apply safe lint fixes, then format.'
    @printf '  %-26s %s\n' '    lint-fix-unsafe' '# Apply unsafe lint fixes, then format.'
    @printf '  %-26s %s\n' '    infer'           '# Infer annotations and imports, then apply safe fixes.'
    @printf '  %-26s %s\n' '    upgrade'         '# Upgrade all locked dependencies and synchronize the environment.'

# --- Read-only checks ---

# Check formatting without modifying files.
format *args:
    uv format --check {{ args }}

# Run Ruff lint checks.
lint:
    uv run --locked ruff check

# Run Pyrefly type checks.
typecheck:
    uv run --locked pyrefly check --min-severity info --remove-unused-ignores=all
    uv check --locked

# Run complexity checks.
complexity:
    uv run --locked complexipy --suggest-refactors

# Run tests, optionally forwarding arguments to pytest.
test *args:
    uv run --locked pytest {{ args }}

# Run every test that is not marked slow -- a quick mid-work check.
test-fast *args:
    uv run --locked pytest -m 'not slow' {{ args }}

# Audit locked dependencies for known vulnerabilities.
audit:
    uv audit --locked

# Run all local, read-only validation.
validate:
    uv sync --locked
    just format
    just lint
    just typecheck
    just complexity
    just test

# Run the full CI validation suite.
ci:
    just validate
    just audit

# --- Documentation ---

# Build the documentation site into site/.
doc-build:
    uv run --no-project python scripts/gen_api_docs.py
    uv run --group docs zensical build --strict

# Serve the documentation locally with live-reload. The API pages are
# generated once at startup; restart after adding or removing a module.
doc-serve:
    uv run --no-project python scripts/gen_api_docs.py
    uv run --group docs zensical serve


# --- Mutable operations (writes) ---

# Apply safe lint fixes, then format.
lint-fix:
    uv sync
    uv run ruff check --fix
    uv format

# Apply unsafe lint fixes, then format.
lint-fix-unsafe:
    uv sync
    uv run ruff check --fix --unsafe-fixes
    uv format

# Infer annotations and imports, then apply safe fixes.
infer:
    uv sync
    uv run pyrefly infer --return-types --parameter-types --imports --containers
    uv check --fix
    just lint-fix

# Upgrade all locked dependencies and synchronize the environment.
upgrade:
    uv lock --upgrade
    uv sync
