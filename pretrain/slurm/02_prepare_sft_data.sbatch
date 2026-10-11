#!/bin/bash
#SBATCH --job-name=ssa-sft-data
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# CPU-only (no GPU): builds the mathematics SFT mixture (train.jsonl / val.jsonl in LitGPT JSON format) from the five
# datasets released with Liu et al. (2025), "Understanding R1-Zero-Like Training" (datasets/train of their repository:
# asdiv_2k, gsm_8k, math_12k, math_lvl3to5_8k, orz_57k, each a Hugging Face `save_to_disk` directory). Set SFT_SOURCES
# to the directory that contains those five sub-directories. Seed 1337 and a 5% validation split reproduce the
# 84,006-example training set of the paper. Runs in a minute (the sources total about 100 MB of text).
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/../../ssa/slurm/common.sh}"
cd $PROJECT_ROOT
python pretrain/prepare_sft_data.py --sft-data-path ${SFT_SOURCES:?set SFT_SOURCES to the datasets/train directory of understand-r1-zero} \
  --out-dir $SFT_DATA --seed 1337 --val-split-fraction 0.05
cat $SFT_DATA/metadata.json
echo DONE
