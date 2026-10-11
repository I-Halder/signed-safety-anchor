#!/bin/bash
#SBATCH --job-name=ssa-sft
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# Mathematics SFT (2 epochs, global batch 128) on the final checkpoint of an arm with the anchor frozen
# (kappa=0, sigma=0, s=+1 on every document), then the post-SFT evaluation. One GPU is enough: measured 15 min
# on one A100-80GB with 1.8 GB of host and 60 GB of device memory (40 GB cards need --train.micro_batch_size 4); the
# global batch is preserved by gradient accumulation. Exports: ARM, MODE, optional TOKEN_COND=1, NTASKS, DETERMINISTIC=1, SDPA_MATH=1.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
RUN=$SSA_ROOT/runs/$ARM
LAST=$(ls -d $RUN/phase_0*/ | sort | tail -1)
OUT=$SSA_ROOT/runs/${ARM}_sft
mkdir -p $OUT
if [ "${TOKEN_COND:-0}" = 1 ]; then
  # control-token arm: no anchor, and the learned control-token embedding must NOT be zeroed
  echo '{"enabled": false, "zero_label_rows": false}' > $OUT/asa_config.json
  ASA_EVAL_ARG="--asa-config $OUT/asa_config.json"
elif [ "${MODE:-}" = "off" ]; then
  echo '{"enabled": false}' > $OUT/asa_config.json; ASA_EVAL_ARG=""
else
  python - <<PY
import json; c=json.load(open("$LAST/asa_config.json")); c.update({"kappa": 0.0, "sigma": 0.0, "rademacher_unlabelled": False, "eval_mode": False, "force_sign": None, "unlabelled_negative": False, "plasticity_all": False, "passage_positive": False}); json.dump(c, open("$OUT/asa_config.json","w"), indent=1); print(c)
PY
  ASA_EVAL_ARG="--asa-config $OUT/asa_config.json"
fi
if [ "${DETERMINISTIC:-0}" = 1 ]; then export SSA_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
if [ "${SDPA_MATH:-0}" = 1 ]; then export SSA_SDPA_MATH=1; fi
cd $LITGPT_DIR
ASA_CONFIG=$OUT/asa_config.json SFT_MIDTRAIN_VAL_PATH=$DATA_ROOT/val_midtrain \
srun --ntasks=${NTASKS:-1} python $SSA_DIR/run_litgpt.py finetune_full $LAST/final --data JSON --precision bf16-true \
  --data.json_path $SFT_DATA --data.mask_prompt true --data.prompt_style default --data.num_workers 4 \
  --train.max_seq_length 2048 --train.epochs 2 --train.micro_batch_size 8 --train.global_batch_size 128 --train.lr_warmup_fraction 0.1 --train.max_lr 0.0001 \
  --train.save_interval 1000 --eval.interval 500 --eval.max_iters 100 --train.log_interval 10 --devices ${NTASKS:-1} --logger_name csv \
  --out_dir $OUT --seed 1337
cd $PROJECT_ROOT
DEV0=${CUDA_VISIBLE_DEVICES:-0}; DEV0=${DEV0%%,*}   # evaluate on the first visible device
CUDA_VISIBLE_DEVICES=$DEV0 python $SSA_DIR/evaluate_checkpoint.py --checkpoint $OUT/final $ASA_EVAL_ARG --pairs $DATA_ROOT/heldout_pairs.jsonl \
  --val-midtrain $DATA_ROOT/val_midtrain --val-pretrain $DATA_ROOT/val_pretrain --tokenizer-dir $TOK_DIR --n-val-batches 40 --out $OUT/eval_post_sft.json
echo SFT_DONE $ARM
