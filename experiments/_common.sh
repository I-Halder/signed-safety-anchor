#!/bin/bash
# Shared helper for the per-experiment scripts in this directory. It is sourced, not executed.
#
# Every experiment script submits three SLURM jobs, each starting only after the previous one succeeded:
#   1. midtraining: 36 phases from the phase-15 checkpoint on NGPUS GPUs of one node   (ssa/slurm/03_arm.sbatch)
#   2. two epochs of mathematics SFT on one GPU, then the post-SFT evaluation           (ssa/slurm/04_sft_eval.sbatch)
#   3. free generations after midtraining and after SFT, one GPU per stage (Figure 3)   (ssa/slurm/09_generated_behavior.sbatch)
#
# Cluster settings (environment variables; sbatch itself reads SBATCH_ACCOUNT):
#   SBATCH_ACCOUNT   your SLURM account, if your cluster requires one
#   GPU_PARTITION    a partition whose nodes have at least NGPUS GPUs        (default: gpu)
#   NGPUS            GPUs (= ranks) for midtraining                           (default: 4, as in Table 1)
#
# NGPUS fixes the order in which the data are read (litdata assigns data to ranks x 8 loader workers), so keep it at 4
# to reproduce the data stream of Table 1. All seeds are pinned (README, "Seeds"):
#   training seed 42 (query noise, random signs of unlabelled documents), data seeds 42 (litdata shard shuffles and
#   mixture sampler), anchor seed 20260921 (direction u and initial functional a), SFT seed 1337, generation seed
#   20260924 combined with (experiment, stage, condition).
set -euo pipefail
EXP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$EXP_DIR/.." && pwd)
SLURM_DIR=$REPO/ssa/slurm
export SSA_COMMON=$SLURM_DIR/common.sh
source "$SSA_COMMON"                       # activates the environment and sets SSA_ROOT, DATA_ROOT, INIT_CKPT, ...
NG=${NGPUS:-4}
PART=${GPU_PARTITION:-gpu}
mkdir -p "$SSA_ROOT/logs" "$SSA_ROOT/generated_behavior"

# submit_experiment <experiment id> <run directory> <anchor mode> <extra exports> <controller flags>
#   anchor mode: off (no anchor), unsigned (s=+1 on every document), fixed (signed anchor, fixed gain)
submit_experiment() {
  local EID=$1 ARM=$2 MODE=$3 EXTRA=$4 CTRL=$5
  local COMMON="ALL,ARM=$ARM,MODE=$MODE,TRAIN_SEED=42${EXTRA:+,$EXTRA}"
  # Midtraining. Measured: 3 h 47 min on 4x A100-80GB, 2 h 19 min on 4x H200; at most 2.5 GB host memory per rank.
  local J1
  J1=$(sbatch --parsable --partition="$PART" --nodes=1 --gres=gpu:$NG --ntasks-per-node=$NG --cpus-per-task=12 \
        --mem=$(( 4 * NG ))G --time=05:00:00 --requeue --open-mode=append --job-name="ssa-$EID" \
        --output="$SSA_ROOT/logs/%x-%j.out" \
        --export="$COMMON,W_START=0.9,W_END=0.1,NTASKS=$NG${CTRL:+,CTRL_ARGS=$CTRL}" "$SLURM_DIR/03_arm.sbatch")
  # SFT + post-SFT evaluation. Measured: 9-15 min on one GPU, 1.9 GB host memory.
  local J2
  J2=$(sbatch --parsable --partition="$PART" --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=12 \
        --mem=4G --time=00:45:00 --requeue --open-mode=append --dependency=afterok:$J1 --job-name="ssa-sft-$EID" \
        --output="$SSA_ROOT/logs/%x-%j.out" --export="$COMMON,NTASKS=1" "$SLURM_DIR/04_sft_eval.sbatch")
  # Generations for Figure 3 and column R: task 0 and task 1 take the two (arm, stage) jobs of this experiment.
  # Measured: 4-17 min per stage on one GPU, 1.2-2 GB host memory.
  local J3
  J3=$(sbatch --parsable --partition="$PART" --array=0-1 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 \
        --mem=4G --time=00:45:00 --requeue --open-mode=append --dependency=afterok:$J2 --job-name="ssa-gen-$EID" \
        --output="$SSA_ROOT/logs/%x-%A_%a.out" \
        --export="ALL,GB_ARGS=--arms $EID,GB_OUT_DIR=$SSA_ROOT/generated_behavior/$EID" "$SLURM_DIR/09_generated_behavior.sbatch")
  echo "$EID ($ARM, mode=$MODE${EXTRA:+, $EXTRA}${CTRL:+, $CTRL}): midtraining $J1 -> SFT+eval $J2 -> generations $J3"
}
