#!/bin/bash
#SBATCH --job-name=ssa-dataprep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# CPU-only job (no GPU): tokenises the midtraining corpora into litdata shards from LOCAL snapshots of the two
# Hugging Face datasets (compute nodes may have no network access; download the snapshots on a login node first,
# see README). Measured: 51 min and 38 GB of host memory with 32 cores.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
cd $PROJECT_ROOT
mkdir -p $DATA_ROOT/train
if [ ! -f $DATA_ROOT/geodesic_meta.json ]; then
  echo "=== geodesic side: passages (train/w1, val_midtrain), labelled shards (train/y1, train/z1), held-out pairs ==="
  python $SSA_DIR/prepare_geodesic_asa.py --dataset $HF_GEODESIC_LOCAL --out-root $DATA_ROOT --tokenizer-dir $TOK_DIR --num-workers 16
else
  echo "geodesic shards already present, skipping"
fi
echo "=== fineweb-edu main shard (4.5M docs -> train/1) and validation shard (50k docs -> val_pretrain) ==="
rm -rf $DATA_ROOT/train/1 $DATA_ROOT/val_pretrain
python $SSA_DIR/prepare_fineweb_asa.py --parquet-dir $HF_FINEWEB_LOCAL --out-root $DATA_ROOT --tokenizer-dir $TOK_DIR --num-workers 32
ls -la $DATA_ROOT $DATA_ROOT/train; du -sh $DATA_ROOT/train/* $DATA_ROOT/val_*
echo DONE
