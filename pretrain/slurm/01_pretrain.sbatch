#!/bin/bash
#SBATCH --job-name=ssa-pretrain
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Pretraining: Pythia-160M from scratch on 3.2B tokens of FineWeb-Edu, as 16 phases of 200M tokens under the first
# part of ONE cosine learning-rate schedule (3e-4 -> 3e-5) laid over the full 9B-token budget. The midtraining arms
# resume this schedule from phase 15, so the plan must be generated with the same token budget and phase counts as
# ssa/run_arm.py uses (16 pretraining + 36 midtraining phases, 3.2B / 9.0B tokens); only phases 000-015 are run here.
# Measured: the complete 52-phase, 9B-token run took 6 h on 4x A100-40GB (2.6 GB of host memory per rank, 39.7 GB
# of device memory per GPU), so the 16 pretraining phases (3.2B tokens, 36% of the steps) should take about
# 2 h 10 min. Afterwards the phase-15 checkpoint is converted into the anchored initial checkpoint used by every
# anchored arm.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/../../ssa/slurm/common.sh}"
cd $PROJECT_ROOT
python pretrain/run_scheduled_phases.py \
  --base-config configs/pythia_160m_pretrain.yaml --run-name fineweb-pretrain --output-root $SSA_ROOT/pretrain \
  --litgpt-dir $LITGPT_DIR --tokenizer-dir $TOK_DIR \
  --data-path $PRETRAIN_DATA_ROOT/train --val-data-path $PRETRAIN_DATA_ROOT/val_pretrain \
  --midtrain-start-token-count 3200000000 --total-token-count 9000000000 --num-pretrain-phases 16 --num-midtrain-phases 36 \
  --main-key main --midtrain-key web --mix-mode weighted --data-start-fraction 0.9 --data-end-fraction 0.1 \
  --data-scheduler linear --pretrain-lr-scheduler cosine --continuous-lr --max-lr 3e-4 --min-lr 3e-5 --seq-length 2048 \
  --launcher-prefix srun --generate-only
cd $LITGPT_DIR
for cfg in $(ls $PRETRAIN_RUN/phase_configs/phase_0*_pretrain_*.yaml | sort); do
  name=$(basename $cfg .yaml)
  if [ -f $PRETRAIN_RUN/$name/final/lit_model.pth ]; then echo "skip $name (done)"; continue; fi
  echo "=== $name ==="
  srun --ntasks=4 python -m litgpt pretrain --config $cfg
done
cd $PROJECT_ROOT
python ssa/convert_checkpoint.py --src $INIT_CKPT_PLAIN --dst $INIT_CKPT --seed 20260921
echo PRETRAIN_DONE
