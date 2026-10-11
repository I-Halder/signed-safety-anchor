#!/bin/bash
#SBATCH --job-name=ssa-verify-controls
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# CPU ONLY (no GPU). Gate for the shuffled-sign (E5) and control-token (E6) arms: exits non-zero if either is
# misconfigured, so their GPU jobs can depend on it with --dependency=afterok. Loads the 160M checkpoint (1 GB) once.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
python $SSA_DIR/tests/verify_controls.py $INIT_CKPT
