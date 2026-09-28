# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "numpy>=2.5.2",
#     "tensorflow[and-cuda]; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "tensorflow; sys_platform == 'darwin'",
# ]
# ///
"""The TensorFlow half of `gpu_coexistence.py`. Never imported -- only `uv run`.

This file deliberately has no `deconvolve` import and is not reachable from the
package: it runs under Python 3.13 with the TensorFlow Keras backend, which is
exactly the environment `src/deconvolve` cannot coexist with. The PEP 723 header above
is the whole quarantine mechanism; `uv run --no-project` provisions it.

It answers one question -- **can TensorFlow get usable GPU memory right now?**
-- and is careful to separate three outcomes that a naive probe collapses into
one:

* `ok`            TF ran the op on the GPU. The coexistence works.
* `cpu_fallback`  TF ran, but on the CPU, because it saw no GPU at all. This is
                  the dangerous outcome: nothing raises, the baseline just runs
                  ~50x slower and the result looks fine. An explicit
                  `tf.device("/GPU:0")` is what turns it into an error, so the
                  probe pins the device rather than letting TF place the op.
* `oom`           TF saw the GPU and could not get memory on it. This is the
                  failure the whole benchmark exists to detect.

The op is a real matmul rather than a device query because TF's memory pool is
lazy about the *first* allocation but greedy after it: listing devices succeeds
on a card that has nothing left to give. Only touching it settles the question.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

# Deliberately verbose, and it was not always: this started at "3" (fatal only),
# which is the setting everyone copies to quieten TF's startup banner. It also
# suppresses the `Could not load dynamic library` and
# `failed call to cuInit` lines, which are the *only* place TensorFlow ever says
# why it decided there is no GPU. A probe whose entire job is to explain a
# missing GPU must not silence the explanation. The cost is nothing: TF logs to
# stderr and this script's protocol is one line of JSON on stdout, so the driver
# can keep the noise and still parse the result.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
# `Cannot dlopen some GPU libraries ... mentioned above` is TF's summary; the
# line naming the library is emitted by dso_loader at VLOG(1), which
# TF_CPP_MIN_LOG_LEVEL does not reach. Scoped to the one module so stderr stays
# readable -- a global TF_CPP_MAX_VLOG_LEVEL buries it again.
os.environ.setdefault("TF_CPP_VMODULE", "dso_loader=1")

# The size is chosen to need real memory (~3 x 512MB of float32) without being
# a meaningful fraction of a 40GB card on its own: a failure here is a failure
# to get a *foothold*, not a failure to fit a large model.
SIDE: int = 8192
ALLOCATION_BYTES: int = SIDE * SIDE * 4


def _peak_bytes() -> int | None:
    """Peak GPU bytes TF's allocator reached, if it can report them."""
    import tensorflow as tf

    try:
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except (KeyError, ValueError, RuntimeError):
        return None


# The sonames a CUDA 12 build of TensorFlow actually dlopens. Hard-coded rather
# than discovered, because the question is whether the loader can resolve the
# exact names TF asks for -- not whether some CUDA exists on the node.
_TF_CUDA_SONAMES: tuple[str, ...] = (
    "libcuda.so.1",
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libnccl.so.2",
    "libnvJitLink.so.12",
    "libcupti.so.12",
)


def _dlopen_report() -> dict[str, str]:
    """Try each soname TF needs and record what the loader said.

    This exists because TF's own account of the failure is a summary that
    refers to lines it may not have printed. Asking the dynamic loader directly
    is deterministic, needs no log-level coaxing, and names every missing
    library at once rather than the first one TF happened to give up on.
    """
    import ctypes

    report: dict[str, str] = {}
    for soname in _TF_CUDA_SONAMES:
        try:
            _ = ctypes.CDLL(soname)
        except OSError as exc:
            report[soname] = f"FAIL: {str(exc)[:160]}"
        else:
            report[soname] = "ok"
    return report


def _nvidia_wheel_libs() -> list[str]:
    """The `.so` files the nvidia pip wheels actually put on disk.

    `nvidia_packages` says the wheels are installed; this says what sonames they
    provide. The two disagree exactly when the installed CUDA minor version
    ships a soname TF was not built against, which is invisible from the
    package list alone.
    """
    try:
        import nvidia
    except ImportError:
        return []

    names: set[str] = set()
    for root in nvidia.__path__:
        names.update(path.name for path in Path(root).glob("*/lib/*.so*"))
    return sorted(names)[:60]


def _environment(dlopen: dict[str, str]) -> dict[str, object]:
    """Everything that can explain a missing GPU, gathered whether or not one is.

    Each entry here is a distinct way the worker can end up on the CPU on a node
    that demonstrably has four A100s, and they are not distinguishable from the
    outside:

    * `built_with_cuda` false means uv installed the CPU-only `tensorflow`
      wheel -- the `[and-cuda]` extra's marker did not match -- and no amount of
      driver or allocation work will help.
    * `nvidia_packages` empty with `built_with_cuda` true means the CUDA wheels
      are absent even though TF expects them.
    * `cuda_visible_devices` of `""` is a masked GPU, which SLURM does to a step
      that did not request one. Absent entirely is different and usually fine.
    * `ld_library_path` matters because a system CUDA ahead of the pip wheels on
      the path is how TF ends up loading a `libcudart` it was not built against.

    `dlopen` is passed in rather than measured here, and the distinction is not
    cosmetic: `ctypes.CDLL` on a library TensorFlow has already loaded succeeds
    because it is resident in the process, so a report taken after the TF import
    says every library resolved no matter what the loader did on the first
    attempt. The snapshot must be taken before TF is imported.
    """
    from importlib import metadata

    import tensorflow as tf

    packages = sorted(
        name
        for dist in metadata.distributions()
        if (name := dist.metadata["Name"] or "").startswith(("nvidia-", "tensorflow"))
    )
    return {
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "all_devices": [d.name for d in tf.config.list_physical_devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "slurm_gpu_vars": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith("SLURM") and "GPU" in k.upper()
        },
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "<unset>")[:600],
        "nvidia_packages": packages,
        "wheels_preloaded": os.environ.get("_RAN_PROBE_REEXEC") == "1",
        "dlopen": dlopen,
        "nvidia_wheel_libs": _nvidia_wheel_libs(),
    }


def probe(dlopen: dict[str, str]) -> dict[str, object]:
    import numpy as np
    import tensorflow as tf

    result: dict[str, object] = {
        "tf_version": tf.__version__,
        "python": sys.version.split()[0],
        "requested_bytes": ALLOCATION_BYTES,
        # Attached on success as well as failure. A probe that only explains
        # failures cannot tell anyone which configuration to reproduce, and the
        # passing configuration is the one that has to reach `submit.sh`.
        "environment": _environment(dlopen),
    }

    gpus = tf.config.list_physical_devices("GPU")
    result["gpus_visible"] = [d.name for d in gpus]

    if not gpus:
        # Not an error on a laptop; the driver decides whether it is one here.
        result["status"] = "cpu_fallback"
        result["detail"] = "tf.config.list_physical_devices('GPU') returned nothing"
        return result

    try:
        # Pinned, not placed: soft placement would silently fall back to CPU on
        # OOM and report success.
        with tf.device("/GPU:0"):
            a = tf.random.normal((SIDE, SIDE), dtype=tf.float32)
            b = tf.random.normal((SIDE, SIDE), dtype=tf.float32)
            c = tf.linalg.matmul(a, b)
            checksum = float(tf.reduce_sum(c).numpy())
        result["status"] = "ok"
        result["device_used"] = c.device
        result["checksum_finite"] = bool(np.isfinite(checksum))
        result["peak_bytes"] = _peak_bytes()
    except tf.errors.ResourceExhaustedError as exc:
        result["status"] = "oom"
        result["detail"] = str(exc).splitlines()[0][:400]
    except (tf.errors.InternalError, tf.errors.UnknownError, RuntimeError) as exc:
        # A CUDA context that cannot be created at all lands here rather than in
        # ResourceExhaustedError, and on a full card it is the common shape.
        result["status"] = "oom"
        result["detail"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}"
    return result


def _wheel_lib_dirs() -> list[str]:
    """The `lib` directory of every installed nvidia pip wheel."""
    try:
        import nvidia
    except ImportError:
        return []

    dirs: list[str] = []
    for root in nvidia.__path__:
        dirs.extend(
            str(path) for path in sorted(Path(root).glob("*/lib")) if path.is_dir()
        )
    return dirs


def _preload_wheels_and_reexec() -> None:
    """Put the pip CUDA wheels ahead of the system CUDA, then start over.

    The hypothesis this tests: NERSC's default environment puts a CUDA 13 tree
    on `LD_LIBRARY_PATH`, and a CUDA 12 build of TensorFlow searching that path
    first finds a `libcublas` and friends it cannot use. The pip `-cu12` wheels
    are installed and correct; they are simply not what the loader reaches
    first.

    The re-exec is not avoidable. `LD_LIBRARY_PATH` is read by the dynamic
    loader when the process starts, so rewriting it inside a running
    interpreter changes nothing for libraries TF has yet to open -- a detail
    that makes an in-process "fix" look like it works while measuring the
    unfixed path. `_DECONVOLVE_PROBE_REEXEC` guards against looping.
    """
    if os.environ.get(key="DECONVOLVE_PROBE_PRELOAD_WHEELS") != "1":
        return
    if os.environ.get(key="_DECONVOLVE_PROBE_REEXEC") == "1":
        return

    dirs: list[str] = _wheel_lib_dirs()
    if not dirs:
        return

    existing: str = os.environ.get(key="LD_LIBRARY_PATH", default="")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join([*dirs, existing]) if existing else ":".join(dirs)
    )
    os.environ["_DECONVOLVE_PROBE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])  # ruff: ignore[start-process-with-no-shell]


def main() -> None:
    _preload_wheels_and_reexec()
    # Before any TensorFlow import, for the reason given in `_environment`.
    dlopen: dict[str, str] = _dlopen_report()
    try:
        result: dict[str, object] = probe(dlopen)
    # The driver needs a reason on stdout, not a traceback on stderr: an
    # unparseable worker is indistinguishable from a crashed one.
    except BaseException as exc:  # ruff: ignore[blind-except]
        result = {
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    # One line of JSON on stdout is the entire protocol. TF writes banners to
    # stderr regardless of TF_CPP_MIN_LOG_LEVEL, so stdout must stay clean.
    print(json.dumps(obj=result))


if __name__ == "__main__":
    main()
