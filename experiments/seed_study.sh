#!/bin/bash
# Seed study of Section 5.3 and the determinism check quoted in the caption of Table 1 (not one of the ten rows).
# E2 at training seed 42 and E4 at training seeds 42, 43 and 44, with deterministic kernels (torch deterministic
# algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, math attention backend instead of flash attention), so that a rerun on
# the same GPU model and layout is bitwise identical. Everything else is pinned: anchor seed 20260921 (the direction u
# and the initial functional a = W_K^T u are the same for every seed), data seeds 42, SFT seed 1337. Only the training
# seed varies, which draws the query noise xi and the random signs of unlabelled documents (E2 draws nothing from it).
#
# Usage:  bash experiments/seed_study.sh            # 4 GPUs, the layout of Table 1 (the paper ran it on 4x H200)
#         NGPUS=2 bash experiments/seed_study.sh    # 2 GPUs, the second layout of Section 5.3 (2x H200)
# Then:   python analysis/seed_study.py --run-root $SSA_ROOT/runs
#
# Run directories: E02_det<N>g_s42, E04_det<N>g_s{42,43,44} (+ _sft). No generations: Section 5.3 is likelihood only.
# Measured on H200 with deterministic kernels: midtraining 3 h 41 min - 3 h 55 min on 4 GPUs, 6 h 38 min - 7 h 08 min on
# 2 GPUs (the math attention backend roughly doubles the step time); at most 2.6 GB host memory per rank; 55 GB device
# memory per GPU, so 80 GB cards are needed at micro-batch 16. SFT 18-23 min on one GPU, 1.9 GB host memory.
source "$(dirname "$0")/_common.sh"
case $NG in
  4) ARM_TIME=05:30:00 ;;
  2) ARM_TIME=08:30:00 ;;
  *) ARM_TIME=${ARM_TIME:?set ARM_TIME (HH:MM:SS) for NGPUS=$NG} ;;
esac

submit_seed_run() {
  local RUN=$1 MODE=$2 SEED=$3
  local COMMON="ALL,ARM=$RUN,MODE=$MODE,TRAIN_SEED=$SEED,DETERMINISTIC=1,SDPA_MATH=1"
  local J1 J2
  J1=$(sbatch --parsable --partition="$PART" --nodes=1 --gres=gpu:$NG --ntasks-per-node=$NG --cpus-per-task=12 \
        --mem=$(( 4 * NG ))G --time=$ARM_TIME --requeue --open-mode=append --job-name="ssa-seed-$RUN" \
        --output="$SSA_ROOT/logs/%x-%j.out" \
        --export="$COMMON,W_START=0.9,W_END=0.1,NTASKS=$NG" "$SLURM_DIR/03_arm.sbatch")
  J2=$(sbatch --parsable --partition="$PART" --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=12 \
        --mem=4G --time=00:45:00 --requeue --open-mode=append --dependency=afterok:$J1 --job-name="ssa-seed-sft-$RUN" \
        --output="$SSA_ROOT/logs/%x-%j.out" --export="$COMMON,NTASKS=1" "$SLURM_DIR/04_sft_eval.sbatch")
  echo "$RUN (mode=$MODE, training seed $SEED, deterministic, $NG GPUs): midtraining $J1 -> SFT+eval $J2"
}

submit_seed_run E02_det${NG}g_s42 off 42
for SEED in 42 43 44; do
  submit_seed_run E04_det${NG}g_s$SEED fixed $SEED
done
