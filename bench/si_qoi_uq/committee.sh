#!/bin/bash
# committee.sh <samples.npz> <nproc> [--mean-first] [--record]: the harness for every sample, split over nproc
# processes (4 threads each), on this node.  Run detached on lestrade/mnf148.
set -uo pipefail
S=$1; P=$2
cd "$(dirname "$0")/../.."
W=$PWD
LOG=${SIQ_RUNS:-/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq}/logs
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=$W:$W/src
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
PY=/storage/eng/essswb/venvs/si-tests/bin/python
tag=$(basename "$S" .npz)
if [ "${3:-}" = "--mean-first" ]; then
  $PY -m bench.si_qoi_uq.harness run "$S" --mean > "$LOG/committee_${tag}_mean.log" 2>&1
fi
K=$($PY -c "import numpy as np; print(len(np.load('$S')['theta']))")
for i in $(seq 0 $((P - 1))); do
  ks=$(seq -s, $i $P $((K - 1)))
  $PY -m bench.si_qoi_uq.harness run "$S" --ks "$ks" ${SIQ_RECORD:+--record} --times "$LOG/committee_${tag}_$i.times.json" \
    > "$LOG/committee_${tag}_$i.log" 2>&1 &
done
wait
echo "committee $tag done"
