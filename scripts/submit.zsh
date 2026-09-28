#!/usr/bin/env zsh
#SBATCH -qshared
#SBATCH -Cgpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -G1
#SBATCH -c32

# The run takes about 4 minutes in total. 10 gives ~2.5x margin.
#SBATCH -t00:10:00

#SBATCH -Jran_e2e
#SBATCH -oslurm-%j.log

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/deconvolve
cd "${PROJECT_DIR}"

echo "DECONVOLVE_CACHE_DIR = ${DECONVOLVE_CACHE_DIR:-<unset: using ./.cache>}"

# Report where the wall clock times.
export DECONVOLVE_TIMING=1

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

TRAIN_ARGS=(-Djets "-n${N_SAMPLES}" -l3 -u128)

mkdir -p runs
marker="$(mktemp)"
trap 'rm -f "${marker}"' EXIT

uv run deconvolve train "${TRAIN_ARGS[@]}" "$@"

RUN_DIR=( runs/*(/e:'[[ $REPLY -nt $marker ]]':) )
RUN_DIR=$RUN_DIR[-1]

echo "Run dir: ${RUN_DIR}"

uv run deconvolve baseline ibu "${RUN_DIR}"
uv run deconvolve train --load-run "${RUN_DIR}"
uv run deconvolve evaluate "${RUN_DIR}" --force

if (( ! $+commands[pdflatex] )); then
  source "${PROJECT_DIR}/scripts/_lmod.zsh"
  module load texlive
fi
uv run deconvolve report "${RUN_DIR}"

echo "Artifacts in ${RUN_DIR}:"
ls -1 "${RUN_DIR}"

jax_cache="${DECONVOLVE_CACHE_DIR:-.cache}/jax"
echo "XLA cache: $(find "${jax_cache}" -type f 2>/dev/null | wc -l | tr -d ' ') entries in ${jax_cache}"
