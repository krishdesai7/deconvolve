#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/deconvolve
SEED_START=200
SEED_END=271

NODES=1
GPUS_PER_NODE=4

RUN_DIR="${PROJECT_DIR}/runs/precision_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${RUN_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Seeds ${SEED_START}-${SEED_END} (72 paired seeds, 144 runs) on ${GPUS_PER_NODE} GPUs"

# `--gpus-per-node` has no short form, and `-G` is NOT it: `-G/--gpus` is the
# TOTAL across the allocation. Shortened to `-G ${GPUS_PER_NODE}` this asks for
# 4 GPUs spread over ${NODES} nodes, which sbatch rejects outright once NODES > 4
# ("Failed to validate job spec, --gpus < -N") and, for NODES <= 4, silently
# grants one GPU per node instead of four -- the sweep still finishes, in four
# times the waves. Leave it long.
JOB=$(sbatch --parsable \
  -qregular -Cgpu -Am3246_g -t00:45:00 \
  -N "${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  -Jprecision_bench -o "${RUN_DIR}/slurm-%j.log" \
  --export="ALL,RUN_DIR=${RUN_DIR},SEED_START=${SEED_START},SEED_END=${SEED_END},GPUS_PER_NODE=${GPUS_PER_NODE},PROJECT_DIR=${PROJECT_DIR}" \
  <<'EOF'
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"

step="srun --exact -N1 -n1 --gpus-per-task=1 -c16 --mem-per-gpu=56G"

for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  for dtype in float64 float32; do
    $step bash -c "
        uv run benchmarks/precision.py '${dtype}' '${seed}'
      " > "${RUN_DIR}/${dtype}_seed$(printf '%04d' "${seed}").log" 2>&1 &

    while (( $(jobs -rp | wc -l) >= GPUS_PER_NODE )); do
      wait -n || true
    done
    sleep 0.5  # small stagger to avoid "srun: Job step creation temporarily disabled"
  done
done
wait || true  # let the final wave finish (ignore individual failures)

for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  grep -h '^SUMMARY' "${RUN_DIR}/float64_seed$(printf '%04d' "${seed}").log" >> "${PROJECT_DIR}/f64.log" || true
  grep -h '^SUMMARY' "${RUN_DIR}/float32_seed$(printf '%04d' "${seed}").log" >> "${PROJECT_DIR}/f32.log" || true
done

n64=$(grep -c '^SUMMARY' "${PROJECT_DIR}/f64.log" || true)
n32=$(grep -c '^SUMMARY' "${PROJECT_DIR}/f32.log" || true)
echo "f64.log now has ${n64} SUMMARY lines; f32.log now has ${n32}"
EOF
)

echo "Submitted job: ${JOB}"
echo "Logs:    ${RUN_DIR}/slurm-${JOB}.log  (+ per-run <dtype>_seedNNNN.log)"
echo "Results: appended to ${PROJECT_DIR}/f64.log and ${PROJECT_DIR}/f32.log"
