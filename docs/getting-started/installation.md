<!-- markdownlint-disable-file no-inline-html -->

# Installation

<span style="font-variant: small-caps;">Deconvolve</span> installs a single command-line program, `deconvolve`. The CLI can be installed through a Python installer such as `uv`, `pipx` or `pip`.

## Prerequisites

- **Python** `>= 3.12`
- A **Python installer** such as `uv`, `pipx` or `pip`. `uv` can fetch a suitable python interpreter, and therefore does not require one to be available on the system.

## Running <span style="font-variant: small-caps;">Deconvolve</span> without installation

Use [`uvx`](https://docs.astral.sh/uv/guides/tools/) or a similar CLI runner to quickly invoke `deconvolve` in an ephemeral environment without permanent installation.

=== "uvx"

    ```shell
    uvx deconvolve
    ```

=== "pipx"

    ```shell
    pipx run deconvolve
    ```

## Installation methods

### Adding <span style="font-variant: small-caps;">Deconvolve</span> to a project

!!! tip

    Adding `deconvolve` as a dependency ensures that all collaborators on the project are using the same version of the module.

Use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) or a project manager of choice to add `deconvolve` as a project dependency:

=== "uv (recommended)"

    ```shell
    uv add deconvolve
    ```

    Then to invoke `deconvolve`, run:

    ```shell
    uv run deconvolve
    ```

    To update `deconvolve`, use `--upgrade-package`

    ```shell
    uv lock --upgrade-package deconvolve
    ```

=== "pip"

    ```shell
    python -m venv .venv
    source .venv/bin/activate
    pip install deconvolve
    ```

    Then to invoke `deconvolve`, run:

    ```shell
    deconvolve
    ```

    To update `deconvolve`, run:

    ```shell
    pip install -U deconvolve
    ```

    It is recommended that <span style="font-variant: small-caps;">Deconvolve</span> be installed into a virtual environment rather than the system interpreter. <span style="font-variant: small-caps;">Deconvolve</span> requires JAX and Keras, amongst other dependencies, and pinning them system-wide may conflict with other installed tools.

### Installing <span style="font-variant: small-caps;">Deconvolve</span> globally

=== "uv (recommended)"

    ```shell
    uv tool install deconvolve@latest
    ```

    `uv tool install` places `deconvolve` on the system `PATH` in its own isolated environment, so that its dependencies cannot collide with any other installed tools.

    To update `deconvolve`, use `uv tool upgrade`:

    ```shell
    uv tool upgrade deconvolve
    ```

=== "pipx"

    ```shell
    pipx install deconvolve
    ```

    Like `uv tool install`, `pipx install` places `deconvolve` on the system `PATH` in its own isolated environment, so that its dependencies cannot collide with any other installed tools.
    
    To update `deconvolve`, use `pipx upgrade`:

    ```shell
    pipx upgrade deconvolve
    ```

---

## Hardware and accelerator support

The JAX dependency resolves by platform, so the correct build is selected automatically by the installer above.

=== "Linux (x86_64, NVIDIA GPU)"

    <span style="font-variant: small-caps;">Deconvolve</span> installs `jax[cuda13]`, built against CUDA 13.

    The CUDA runtime libraries are provided as PyPI wheels and are resolved automatically, so the only host requirement is an NVIDIA driver new enough for CUDA 13. No system provided CUDA toolkit is required.

=== "macOS (Apple Silicon, arm64)"

    <span style="font-variant: small-caps;">Deconvolve</span> installs plain `jax` and runs on the CPU. The official macOS arm64 JAX wheels provide no GPU acceleration.

    Experimental alternatives, such as `jax-mps` or `IREE`-based workflows, may enable Metal acceleration, but these configurations are not officially tested or supported by <span style="font-variant: small-caps;">Deconvolve</span>. Users should independently validate their correctness and performance.

=== "Other platforms"

    Any other platform resolves to CPU-only JAX if a wheel exists for it. Only the two configurations above are officially supported.

---

## Shell autocompletion

The `deconvolve` CLI is built with [Typer](https://typer.tiangolo.com/) and can install its own autocompletion script:

!!! tip

    You can run `echo $SHELL` to help you determine your shell.

=== "zsh"

    ```shell
    deconvolve --install-completion zsh
    ```

=== "bash"

    ```shell
    deconvolve --install-completion bash
    ```

=== "fish"

    ```shell
    deconvolve --install-completion fish
    ```

Then restart the shell or source the shell config file.

Shell completion requires the `deconvolve` program to be on the `PATH`: completion is registered against the command name, and therefore cannot be activated through `uv run (-m) deconvolve` or `python -m deconvolve`.

---

## Contributing

Working on <span style="font-variant: small-caps;">Deconvolve</span> requires `uv` to be installed in addition to a checkout of the repository.

1. **Clone the repository:**

    ```shell
    git clone https://github.com/krishdesai7/deconvolve.git
    cd deconvolve
    ```

2. **Synchronize dependencies:**

    ```shell
    uv sync
    ```

    This creates a `.venv` with every runtime and development dependency at
    the exact versions in `uv.lock`.

Inside a checkout, CLI help is available through

```shell
uv run deconvolve --help
```
