#!/bin/bash
#SBATCH --job-name=ssa-pretrain-data
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# CPU-only (no GPU). Tokenises FineWeb-Edu (sample-10BT) into litdata shards for the pretraining stage:
#   train/1      the first 9,000,000 documents of a seed-1234 permutation of the 9.67M-row sample
#   val_pretrain the next 50,000 documents of the same permutation
# Needs network access to the Hugging Face hub (the dataset is read through `datasets`, memory-mapped, ~27 GB on disk
# under $HF_HOME). Run on a node with internet access or pre-populate $HF_HOME. Measured: 54 min and 30 GB of host
# memory for the 9M-document shard, 1 min for the validation shard.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/../../ssa/slurm/common.sh}"
cd $PROJECT_ROOT
mkdir -p $PRETRAIN_DATA_ROOT/train
python pretrain/prepare_hf_litdata.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --split train \
  --output-dir $PRETRAIN_DATA_ROOT/train/1 --tokenizer EleutherAI/pythia-70m --text-column text \
  --num-samples 9000000 --block-size 2048 --skip-samples 0 --seed 1234 --num-workers 16
python pretrain/prepare_hf_litdata.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --split train \
  --output-dir $PRETRAIN_DATA_ROOT/val_pretrain --tokenizer EleutherAI/pythia-70m --text-column text \
  --num-samples 50000 --block-size 2048 --skip-samples 9000000 --seed 1234 --num-workers 16
echo DONE
