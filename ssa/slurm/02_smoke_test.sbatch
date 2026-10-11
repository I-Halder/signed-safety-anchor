#!/bin/bash
#SBATCH --job-name=ssa-smoke
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# GPU smoke test on one (small) GPU: converts the phase-15 checkpoint if needed, then runs 2 tiny midtraining
# phases of the label-signed arm, an evaluation, and 2 SFT steps. Run this before submitting the full arms.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
cd $PROJECT_ROOT
nvidia-smi -L || true
if [ ! -f $INIT_CKPT/lit_model.pth ]; then
  python $SSA_DIR/convert_checkpoint.py --src $INIT_CKPT_PLAIN --dst $INIT_CKPT --seed 20260921
fi
python $SSA_DIR/run_arm.py --arm smoke_label_signed --mode fixed --run-root $SSA_ROOT/runs_smoke --data-root $DATA_ROOT \
  --init-ckpt $INIT_CKPT --init-ckpt-plain $INIT_CKPT_PLAIN --tokenizer-dir $TOK_DIR \
  --ntasks 1 --max-phases 2 --phase-steps-override 2 --micro-batch-size 4 --eval-batches 4 \
  --controller-args "--warmup-phases 1"
echo "=== smoke SFT (2 steps) ==="
OUT=$SSA_ROOT/runs_smoke/smoke_label_signed_sft; mkdir -p $OUT
python - <<PY
import json; c=json.load(open("$SSA_ROOT/runs_smoke/smoke_label_signed/phase_002/asa_config.json")); c.update({"kappa": 0.0, "sigma": 0.0, "rademacher_unlabelled": False}); json.dump(c, open("$OUT/asa_config.json","w"), indent=1)
PY
cd $LITGPT_DIR
ASA_CONFIG=$OUT/asa_config.json SFT_MIDTRAIN_VAL_PATH=$DATA_ROOT/val_midtrain \
python $SSA_DIR/run_litgpt.py finetune_full $SSA_ROOT/runs_smoke/smoke_label_signed/phase_002/final --data JSON --precision bf16-true \
  --data.json_path $SFT_DATA --data.mask_prompt true --data.prompt_style default --data.num_workers 2 \
  --train.max_seq_length 2048 --train.epochs 1 --train.max_steps 2 --train.micro_batch_size 2 --train.global_batch_size 4 --train.lr_warmup_steps 1 --train.max_lr 0.0001 \
  --train.save_interval 1000 --eval.interval 1 --eval.max_iters 2 --train.log_interval 1 --devices 1 --logger_name csv \
  --out_dir $OUT --seed 1337
echo SMOKE_OK
