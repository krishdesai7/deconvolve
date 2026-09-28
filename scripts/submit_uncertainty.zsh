#!/usr/bin/env zsh
#
# Packed launcher for the bootstrap x seed variance design.
#
# One cell is one `deconvolve uncertainty run`, so cells are independent and pack onto
# whatever GPUs the allocation has. Sizing, written out rather than guessed at:
# a cell is one ordinary training run plus the jet cache read. Scaled from a
# measured 12-var/500k/100-epoch run (`timings.json`: epochs 6.3s) by 3.2x the
# rows and ~7x the network for `-u128 -l3`, a 1.6M cell is ~3 min of training
# and ~4 minutes all in. B*S cells over NODES*4 GPUs is ceil(B*S / (NODES*4))
# waves. The default 8x8 on two nodes is 8 waves, ~32 minutes, inside the hour
# requested. Raising NODES cuts waves and buys queue time; it does not make a
# cell faster.
#
# Defaults measure the decomposition. For the bin-to-bin covariance, the
# bootstrap axis needs the replicates, not the seed one -- S can stay at 2
# because the interaction is already measured to 49 df by the 8x8 design:
#
#   B=100 S=2 NODES=4 TIME=01:30:00 zsh scripts/submit_uncertainty.zsh
#
# That is 200 cells over 16 GPUs -- 13 waves, ~52 min at the 1.6M cell above,
# inside the 90 minutes requested.
#
# This is the grid shape the published numbers use
# (notes/agent/uncertainty.md): a B=50 run confirmed against it (every lag
# correlation within 0.01, every effective rank within 0.2) before B=100
# superseded it as the smaller grid's individual off-diagonal entries moved by
# up to 0.39 -- too much to publish a single matrix entry from, even though the
# aggregate structure was already right.
#
# Those published numbers were measured at `-n1000000`. The default below is
# now 1.6M, matching what `scripts/submit.zsh` trains, so a design run here
# supersedes them rather than describing a different model -- see the note in
# `notes/agent/uncertainty.md` under "What the design measured". Both grids
# have to be rerun for that to hold: a decomposition at one sample size and a
# covariance at another do not describe the same measurement.
#
# Warm the jet cache on a login node first; a cold cache pulls 3.3GB from
# Zenodo inside the job.

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/deconvolve

B=${B:-8}
S=${S:-8}
N_EVAL=${N_EVAL:-100000}
N_BINS=${N_BINS:-20}
CELLS=$((B * S))

NODES=${NODES:-2}
# Bump with the grid: the default 8x8 on two nodes is 8 waves (~35 min), but
# B=100 S=2 on four is 13 waves and would hit a one-hour wall.
TIME=${TIME:-01:00:00}
GPUS_PER_NODE=4
GPUS_TOTAL=$((NODES * GPUS_PER_NODE))

# The design measures RAN as the paper ships it, so these are the paper's
# values and not a cheaper stand-in. A design run at other settings is a
# variance budget for a model nobody is publishing -- which is why this tracks
# `scripts/submit.zsh` and why the two must be changed together. All twelve
# observables, by way of `load_jet_dataset`'s default: `--var` is repeatable
# and would append rather than replace.
RUN_ARGS=${RUN_ARGS:--Djets -n1600000 -l3 -u128}

DESIGN_DIR="${PROJECT_DIR}/runs/unc_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${DESIGN_DIR}"

# Fix the design's settings once, here on the login node, before any cell
# exists. Cells read this file rather than the config layers, so an edit to
# deconvolve.toml while the array is in flight cannot split the design.
uv run deconvolve uncertainty freeze "${DESIGN_DIR}" ${RUN_ARGS} \
    -B "${B}" -S "${S}" --n-eval "${N_EVAL}"

echo "Design dir: ${DESIGN_DIR}"
echo "Running ${B}x${S} = ${CELLS} cells on ${GPUS_TOTAL} GPUs"

# `--gpus-per-node` has no short form, and `-G` is NOT it: `-G/--gpus` is the
# TOTAL across the allocation. See the same note in scripts/submit_hparam.zsh.
JOB=$(sbatch --parsable \
  -qregular -Cgpu -Am3246_g -t"${TIME}" \
  -N "${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  -Jdeconvolve_uncertainty \
  -o "${DESIGN_DIR}/slurm-%j.log" \
  --export="ALL,DESIGN_DIR=${DESIGN_DIR},B=${B},S=${S},CELLS=${CELLS},N_BINS=${N_BINS},GPUS_TOTAL=${GPUS_TOTAL},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

step="srun --exact -n1 -N1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=56G"

for cell in $(seq 0 $((CELLS - 1))); do
  log="${DESIGN_DIR}/cell_$(printf '%04d' "${cell}").log"
  # Settings come from design.json, frozen above, not from these flags: a
  # hand-rerun of one failed cell with a different RUN_ARGS must not be able
  # to diverge from what the rest of the array trained under.
  $step bash -c "
      uv run deconvolve uncertainty run '${cell}' '${DESIGN_DIR}'
    " > "${log}" 2>&1 &

  while (( $(jobs -rp | wc -l) >= GPUS_TOTAL )); do
    wait -n || true
  done
  sleep 0.5
done
wait || true

uv run deconvolve uncertainty collect \
    "${DESIGN_DIR}" -B "${B}" -S "${S}" --n-bins "${N_BINS}"
EOF
)

echo "Submitted packed job: ${JOB}"
echo "Logs:    ${DESIGN_DIR}/slurm-${JOB}.log  (+ per-cell cell_NNNN.log)"
echo "Collect: uv run deconvolve uncertainty collect ${DESIGN_DIR} -B ${B} -S ${S}"
