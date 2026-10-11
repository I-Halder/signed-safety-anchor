#!/bin/bash
#SBATCH --job-name=ssa-genbehav
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%A_%a.out
# Generated-behaviour evaluation, generation half (GPU). Every (arm, stage) job is 6 conditions x 259 scenarios x 4
# samples of <=64 new tokens from a 160M model; the 12-task array shards the (arm, stage) jobs by file
# (jobs[RANK::WORLD]), so there is no collective communication and a preempted task resumes with --resume.
# Measured: 12-17 min per task on an A100-40GB with 1.5-2 GB of host memory; the sampler (batch 96) fills the device
# memory it is given. Scoring and plotting are CPU-only: see 10_score_generated_behavior.sbatch.
# If the partition mixes GPU generations, add a --constraint that excludes hardware newer than the pinned torch build supports.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
export RANK=${SLURM_ARRAY_TASK_ID:-0}
export WORLD_SIZE=${SLURM_ARRAY_TASK_COUNT:-1}
echo "shard $RANK/$WORLD_SIZE on $(hostname), gpu $CUDA_VISIBLE_DEVICES"
cd $PROJECT_ROOT
python analysis/generated_behavior.py --generate --resume ${GB_ARGS:-}
echo GENBEHAV_SHARD_DONE $RANK
