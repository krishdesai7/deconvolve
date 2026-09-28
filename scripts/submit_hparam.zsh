#!/usr/bin/env zsh

set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/deconvolve

FLAG=${FLAG:---lr-g}
LEVELS=${LEVELS:-"3e-5 1e-4 3e-4"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}

# Which nuisance axis the replicates sample, and what the other one is pinned
# to. The default samples initialization at a fixed batch order, which is right
# for *comparing arms* -- it is one axis of difference, not two.
#
# It is the wrong setting for quoting a spread. The first lr_g sweep measured
# the per-epoch criterion curves of eight `--seed` replicates correlating at
# r = +0.88 (at lr_g 3e-5), because all eight saw the identical batch sequence:
# they are near-copies, not independent draws, and the SD they give is smaller
# than an ensemble's for that reason alone. Swap the two to measure the other
# half:
#
#   REPLICATE_FLAG=--data-seed FIXED_ARGS="--seed 0" zsh scripts/submit_hparam.zsh
REPLICATE_FLAG=${REPLICATE_FLAG:---seed}
FIXED_ARGS=${FIXED_ARGS:---data-seed 42}

NODES=${NODES:-6}
GPUS_PER_NODE=4
GPUS_TOTAL=$((NODES * GPUS_PER_NODE))

TRAIN_ARGS="-Djets -n1000000 -l3 -u128 --no-plots ${FIXED_ARGS}"

ARM_DIR="${PROJECT_DIR}/runs/hp_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${ARM_DIR}"
echo "Arm dir: ${ARM_DIR}"
echo "Sweeping ${FLAG} over [${LEVELS}] x ${REPLICATE_FLAG} [${SEEDS}], ${FIXED_ARGS} on ${GPUS_TOTAL} GPUs"

# `--gpus-per-node` has no short form, and `-G` is NOT it: `-G/--gpus` is the
# TOTAL across the allocation. Shortened to `-G ${GPUS_PER_NODE}` this asks for
# 4 GPUs spread over ${NODES} nodes, which sbatch rejects outright once NODES > 4
# ("Failed to validate job spec, --gpus < -N") and, for NODES <= 4, silently
# grants one GPU per node instead of four -- the sweep still finishes, in four
# times the waves. Leave it long.
JOB=$(sbatch --parsable \
  -qregular -Cgpu -Am3246_g -t01:00:00 \
  -N "${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  -Jran_hparam \
  -o "${ARM_DIR}/slurm-%j.log" \
  --export="ALL,ARM_DIR=${ARM_DIR},FLAG=${FLAG},LEVELS=${LEVELS},SEEDS=${SEEDS},GPUS_TOTAL=${GPUS_TOTAL},TRAIN_ARGS=${TRAIN_ARGS},REPLICATE_FLAG=${REPLICATE_FLAG},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

step="srun --exact -n1 -N1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-gpu=56G"

for level in ${LEVELS}; do
  for seed in ${SEEDS}; do
    tag="$(printf '%s_seed%02d' "${level}" "${seed}")"
    run_dir="${ARM_DIR}/${tag}"
    mkdir -p "${run_dir}"
    $step bash -c "
        uv run deconvolve train ${TRAIN_ARGS} \
            ${FLAG}='${level}' ${REPLICATE_FLAG}='${seed}' --run-dir='${run_dir}'
      " > "${run_dir}/train.log" 2>&1 &

    while (( $(jobs -rp | wc -l) >= GPUS_TOTAL )); do
      wait -n || true
    done
    sleep 0.5
  done
done
wait || true

uv run benchmarks/hparam_collect.py --arm-dir "${ARM_DIR}"
EOF
)

echo "Submitted packed job: ${JOB}"
echo "Logs:    ${ARM_DIR}/slurm-${JOB}.log  (+ per-run train.log)"
echo "Collect: uv run benchmarks/hparam_collect.py --arm-dir ${ARM_DIR}"
echo "         add --exclude m to score without the observable §4 shows is"
echo "         limited by a non-universal response rather than by tuning."
