#!/bin/bash
#SBATCH --job-name=ssa-arm
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
# One experimental arm: 36 midtraining phases on 4 GPUs, evaluation after every phase.
# Measured: 3 h 47 min on 4x A100-80GB and 2 h 19 min on 4x H200; at most 2.5 GB of host memory per rank
# (hence --mem=32G) and 39 GB of device memory per GPU with micro-batch 16 (40 GB cards need --micro-batch-size 8).
# Exports (see experiments/_common.sh): ARM, MODE=off|unsigned|fixed, optional W_START/W_END (default 0.9/0.1),
# NTASKS, TRAIN_SEED, ANCHOR_SEED, NO_PAIRS=1, LABELLED_WEIGHT, CTRL_ARGS (extra controller.py flags),
# DETERMINISTIC=1 (deterministic CUDA kernels), SDPA_MATH=1 (math attention backend). Runs that are to be compared bitwise
# must also share the GPU model, NTASKS and the micro-batch size, which together fix the data order.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
cd $PROJECT_ROOT
echo "ARM=$ARM MODE=$MODE node=$(hostname) gpus=$(nvidia-smi -L | wc -l) ntasks=${NTASKS:-4}"
python $SSA_DIR/run_arm.py --arm $ARM --mode $MODE --w-start ${W_START:-0.9} --w-end ${W_END:-0.1} \
  --run-root $SSA_ROOT/runs --data-root $DATA_ROOT --init-ckpt $INIT_CKPT --init-ckpt-plain $INIT_CKPT_PLAIN \
  --tokenizer-dir $TOK_DIR --ntasks ${NTASKS:-4} --eval-batches 40 --labelled-weight ${LABELLED_WEIGHT:-0.005} \
  ${TRAIN_SEED:+--train-seed $TRAIN_SEED} ${ANCHOR_SEED:+--anchor-seed $ANCHOR_SEED} ${NO_PAIRS:+--no-pairs} \
  ${DETERMINISTIC:+--deterministic} ${SDPA_MATH:+--sdpa-math} ${CTRL_ARGS:+--controller-args="$CTRL_ARGS"}
echo ARM_DONE $ARM
