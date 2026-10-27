#!/usr/bin/env zsh
#SBATCH -qshared
#SBATCH -Cgpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -G1
#SBATCH -c32

# Deconvolve's own run is ~4 minutes; OmniFold alone measured ~41 for the
# shipped configuration (1.6M samples, 12 jet observables, 3 iterations, 50
# epochs); the redraw, metrics and report on top are a few more. ~50 in all, so
# an hour is ~1.2x margin.
#SBATCH -t01:00:00

#SBATCH -Jdeconvolve_e2e
#SBATCH -oslurm-%j.log

# The whole pipeline in one job: train -> IBU -> OmniFold -> redraw with both
# baselines -> metrics -> report.
#
# OmniFold's worker has its own uv environment (a PEP 723 script, not in
# `uv.lock`), and first use downloads ~3.5GB of CUDA wheels, which a compute
# node usually cannot do. Warm it on a login node once:
#
#     uv run --no-project src/deconvolve/baselines/_omnifold_worker.py

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/deconvolve
cd "${PROJECT_DIR}"

echo "DECONVOLVE_CACHE_DIR = ${DECONVOLVE_CACHE_DIR:-<unset: using ./.cache>}"

# Report where the wall clock times.
export DECONVOLVE_TIMING=1

# Jets unless the arguments say otherwise. Click keeps the last occurrence of
# an option, so the last dataset flag here is the one `train` will see.
DATASET=jets
for (( i = 1; i <= $#; i++ )); do
  case "${@[i]}" in
    -D|--dataset) DATASET="${@[i+1]:-}"; (( ++i )) ;;
    --dataset=*)  DATASET="${@[i]#--dataset=}" ;;
    -D*)          DATASET="${@[i]#-D}" ;;
  esac
done
echo "dataset = ${DATASET}"

TRAIN_ARGS=("-D${DATASET}")

# The full twelve-observable jet run. A Gaussian run keeps `train`'s own
# defaults (and whatever its --config and flags set): these are sized for jets.
if [[ "${DATASET}" == jets ]]; then
  N_REQUESTED=1600000

  # The Zenodo release holds ~1.6M jets per generator, and `load_jet_dataset`
  # raises rather than truncating when asked for more than is on disk. Read the real
  # count off the cache and clamp.
  N_SAMPLES="$(uv run python - "${N_REQUESTED}" <<'PY'
import sys

import numpy as np

from deconvolve.coretypes.constants import (
    CACHE_DIR,
    CACHE_FILENAMES,
    SUBSTRUCTURE_VARIABLES,
)

requested = int(sys.argv[1])
missing = [
    v for v in SUBSTRUCTURE_VARIABLES
    if not (CACHE_DIR / f"{CACHE_FILENAMES[v]}.npz").exists()
]
if missing:
    # Cold cache: 3.3GB from Zenodo.
    print(f"COLD {' '.join(missing)}", file=sys.stderr)
    print(requested)
    raise SystemExit(0)

path = CACHE_DIR / f"{CACHE_FILENAMES[SUBSTRUCTURE_VARIABLES[0]]}.npz"
with np.load(file=path) as f:
    available = min(len(f["z_true"]), len(f["z_gen"]))
print(min(requested, available))
PY
  )"

  if [[ "${N_SAMPLES}" -lt "${N_REQUESTED}" ]]; then
    echo "Clamped n_samples ${N_REQUESTED} -> ${N_SAMPLES} (all that is on disk)."
  fi
  echo "n_samples = ${N_SAMPLES}"

  TRAIN_ARGS+=("-n${N_SAMPLES}" -l3 -u128)
fi

# `module` is a shell function that only a login shell has; see the comment in
# `scripts/_lmod.zsh`. Sourced unconditionally: OmniFold needs `cudatoolkit`
# whatever else is on PATH.
source "${PROJECT_DIR}/scripts/_lmod.zsh"

# the figures `train` and `plot` draw are typeset by LaTeX (`text.usetex`), and `plotting` asks `kpsewhich` at import whether `cochineal.sty` exists.
if (( ! $+commands[pdflatex] )); then
  module load texlive
fi

mkdir -p runs
marker="$(mktemp)"
trap 'rm -f "${marker}"' EXIT

uv run deconvolve train "${TRAIN_ARGS[@]}" "$@"

RUN_DIR=( runs/*(/e:'[[ $REPLY -nt $marker ]]':) )
RUN_DIR=$RUN_DIR[-1]
# Done with it; the EXIT trap is needed below for the CUDA toolkit.
rm -f "${marker}"
trap - EXIT

echo "Run dir: ${RUN_DIR}"

uv run deconvolve baseline ibu "${RUN_DIR}"

# Without `cudatoolkit/12.9`, TensorFlow cannot reach `libcusolver.so.11`,
# registers no GPU, and runs on the CPU without raising -- correct weights, tens
# of times slower. The package warns when the worker reports a CPU; watch for it
# in the log.
#
# Held only for as long as the worker needs it, and released on the way out
# however this script exits. Not `{ ... } always { ... }`: under `set -e`,
# ERR_EXIT leaves the shell before the `always` block runs. An EXIT trap fires
# on both paths.
module load cudatoolkit/12.9
trap 'module unload cudatoolkit/12.9' EXIT

uv run deconvolve baseline omnifold "${RUN_DIR}"

module unload cudatoolkit/12.9
trap - EXIT

# `plot` reloads the saved generator and redraws the figures with both
# baselines on them: `_load_baseline_weights` picks up whichever
# `*_weights.npz` exist at draw time.
uv run deconvolve plot "${RUN_DIR}"
uv run deconvolve evaluate "${RUN_DIR}" --force
uv run deconvolve report "${RUN_DIR}"

echo "Artifacts in ${RUN_DIR}:"
ls -1 "${RUN_DIR}"

jax_cache="${DECONVOLVE_CACHE_DIR:-.cache}/jax"
echo "XLA cache: $(find "${jax_cache}" -type f 2>/dev/null | wc -l | tr -d ' ') entries in ${jax_cache}"
