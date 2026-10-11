# Shared environment for every SLURM script in this directory and in pretrain/slurm. Sourced, not executed.
#
# Edit the three lines marked EDIT for your cluster. Everything under $SSA_ROOT is large (data shards,
# checkpoints, run directories); put it on fast scratch storage, not in the repository.
module load cuda/12.2.0 2>/dev/null || true                                   # EDIT: your CUDA module, if any
source "${SSA_VENV:-$HOME/venvs/ssa}/bin/activate"                           # EDIT: the Python environment (see README)
export SSA_ROOT="${SSA_ROOT:-/scratch/$USER/ssa}"                             # EDIT: large-output root

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export SSA_DIR=$PROJECT_ROOT/ssa
export LITGPT_DIR=$PROJECT_ROOT/litgpt
export DATA_ROOT=$SSA_ROOT/data                                   # midtraining shards + heldout_pairs.jsonl (ssa/prepare_*.py)
export PRETRAIN_DATA_ROOT=$SSA_ROOT/pretrain_data                 # pretraining shards (pretrain/prepare_hf_litdata.py)
export PRETRAIN_RUN=$SSA_ROOT/pretrain/fineweb-pretrain           # output of pretrain/slurm/01_pretrain.sbatch
export INIT_CKPT_PLAIN=$PRETRAIN_RUN/phase_015_pretrain_015/final # the 3.2B-token checkpoint every arm branches from
export INIT_CKPT=$SSA_ROOT/ckpt/phase15_asa                       # the same checkpoint with anchor functionals added
export SFT_DATA=$SSA_ROOT/sft_data                                # prepared mathematics SFT data (pretrain/prepare_sft_data.py)
export TOK_DIR="${TOK_DIR:-$LITGPT_DIR/checkpoints/EleutherAI/pythia-160m}"   # litgpt download EleutherAI/pythia-160m --tokenizer_only true
export HF_HOME="${HF_HOME:-$SSA_ROOT/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_GEODESIC_LOCAL=$DATA_ROOT/hf_geodesic                   # local snapshot of geodesic-research/synth-scenario-docs-positive-alignment-midtraining
export HF_FINEWEB_LOCAL=$DATA_ROOT/hf_fineweb_edu                 # local snapshot of HuggingFaceFW/fineweb-edu, sample/10BT
export GB_OUT_DIR="${GB_OUT_DIR:-$PROJECT_ROOT/results/generated_behavior}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TORCHINDUCTOR_CACHE_DIR=$SSA_ROOT/torchinductor_cache
export TRITON_CACHE_DIR=$SSA_ROOT/triton_cache
export PYTHONPATH=$LITGPT_DIR:$SSA_DIR:${PYTHONPATH:-}
mkdir -p $SSA_ROOT/logs
