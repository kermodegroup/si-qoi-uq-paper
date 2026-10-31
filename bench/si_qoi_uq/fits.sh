#!/bin/bash
# Spec §1 fits: fits.sh <model: o4d20|o3d12> <variant: cr3|cr3_ell3|kappa|pops>
# Same data, basis and flags as the accuracy-cost fits (runs/accuracy_cost/ace/<model>/cmd.txt), plus the UQ
# flags.  --ard-mode sequential keeps --noise shared (the benchmark's noise ratios; joint mode refits sigma_E/F/V
# by evidence, which on o3d12 gave sigma_E/sigma_F ~ 29 and a force-dominated mean: RECON "Fits").  Seed 0.
# Launch detached on mnf148/lestrade (o4d20 > 10 GB).
set -euo pipefail
M=$1; V=$2
DATA=/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost
OUT=${SIQ_RUNS:-/storage/eng/essswb/projects/ace-jax/runs/si_qoi_uq}/fits/$M/$V
case $M in
  o4d20) BASIS="--order 4 --max-degree 20" ;;
  o3d12) BASIS="--order 3 --max-degree 12" ;;
  *) echo "unknown model $M" >&2; exit 2 ;;
esac
case $V in
  cr3)      UQ="--uq ard --ard-cluster-size inf" ;;      # headline: whole-configuration CR3 clusters (RECON Q3)
  cr3_ell3) UQ="--uq ard" ;;                             # package default (3 r_cut blocks): sensitivity
  kappa)    UQ="--uq ard --ard-variance kappa --ard-cluster-size inf" ;;   # B1: posterior shape, keeps chol
  *) echo "unknown variant $V" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
exec /usr/bin/time -v .venv/bin/aj fit --train $DATA/train.xyz --test $DATA/test.xyz $BASIS --rcut 5.0 \
  --virial-key virial --weights $DATA/weights_acejax.json --e0 lsq --rungs map --m-per-species 0 \
  --noise shared --ard-mode sequential --configs-per-batch 1 --seed 0 $UQ --out "$OUT" > "$OUT/fit.log" 2>&1
