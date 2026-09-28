#!/usr/bin/env zsh
#SBATCH -qshared
#SBATCH -Cgpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -G1
#SBATCH -c32

# The OmniFold half alone measured ~41 minutes for the shipped configuration
# (1.6M samples, 12 jet observables, 3 iterations, 50 epochs). The redraw,
# metrics and report on top are a few minutes. 75 is ~1.5x the whole pipeline
# and still well inside what `shared` backfills.
#SBATCH -t01:15:00

#SBATCH -Jran_omnifold
#SBATCH -oslurm-%j.log

# OmniFold against an existing run directory.
#
#     sbatch scripts/submit_omnifold.zsh runs/2026-09-06T203848Z
#
# The run directory is read for its `config.json` and written for the OmniFold
# artifacts; the RAN training in it is neither repeated nor disturbed. This is a
# separate job from `submit.zsh` rather than a step inside it because OmniFold
# takes longer on its own than that job's entire wall clock.
#
# Two things must be true before this will work, and neither fails loudly on its
# own:
#
#   * `module load cudatoolkit/12.9`, which this script does. Without it
#     TensorFlow cannot reach `libcusolver.so.11`, registers no GPU, and runs on
#     the CPU without raising -- correct weights, tens of times slower. The
#     package warns when the worker reports a CPU; watch for it in the log.
#   * The worker's uv environment must already be resolved. It is not in
#     `uv.lock` -- it is a PEP 723 script -- so first use downloads ~3.5GB of
#     CUDA wheels, which a compute node usually cannot do. Warm it on a login
#     node first:
#
#         uv run --no-project src/ran/baselines/_omnifold_worker.py

set -euo pipefail

RUN_DIR="${1:?usage: sbatch scripts/submit_omnifold.zsh <run_dir>}"

PROJECT_DIR=/global/u1/k/kdesai/deconvolve
cd "${PROJECT_DIR}"

if [[ ! -f "${RUN_DIR}/config.json" ]]; then
  print -u2 "No config.json in ${RUN_DIR} -- that is not a run directory."
  exit 1
fi

echo "DECONVOLVE_CACHE_DIR = ${DECONVOLVE_CACHE_DIR:-<unset: using ./.cache>}"
echo "Run dir: ${RUN_DIR}"

export DECONVOLVE_TIMING=1

# Held only for as long as the worker needs it, and released on the way out
# however this script exits.
#
# The obvious zsh idiom here is `{ ... } always { ... }`, and it is wrong under
# `set -e`: ERR_EXIT leaves the shell before the `always` block runs, so a
# failed unfolding would leave the CUDA 12 toolkit loaded. Measured, not
# assumed. An EXIT trap fires on both paths.
# `module` is a shell function that only a login shell has; see the comment in
# `scripts/_lmod.zsh`. Sourced before the first `module` call, not at the top,
# so the cheap argument checks still fail fast on a machine without Lmod.
source "${PROJECT_DIR}/scripts/_lmod.zsh"

module load cudatoolkit/12.9
trap 'module unload cudatoolkit/12.9' EXIT

uv run deconvolve baseline omnifold "${RUN_DIR}" "${@:2}"

module unload cudatoolkit/12.9
trap - EXIT

# `--load-run` reloads the saved generator instead of training and redraws the
# figures, putting the OmniFold curve on them: `_load_baseline_weights` picks
# up whichever `*_weights.npz` exist at draw time. Then re-score, so
# `metrics.json` and the report agree with the figures.
uv run deconvolve train --load-run "${RUN_DIR}"
uv run deconvolve evaluate "${RUN_DIR}" --force

module load texlive
uv run deconvolve report "${RUN_DIR}" --force

echo "Artifacts in ${RUN_DIR}:"
ls -1 "${RUN_DIR}"
