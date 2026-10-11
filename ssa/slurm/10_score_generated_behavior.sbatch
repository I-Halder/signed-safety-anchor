#!/bin/bash
#SBATCH --job-name=ssa-genscore
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Scoring + figure for the generated-behaviour evaluation. Pure CPU (string matching and a 10k-resample bootstrap
# over 259 scenarios); runs in a few minutes and must not be sent to a GPU partition. Reads only the cached
# generations, so it is re-runnable with a different classifier at no cost.
set -euo pipefail
source "${SSA_COMMON:-$SLURM_SUBMIT_DIR/common.sh}"   # submit from ssa/slurm, or export SSA_COMMON=/path/to/common.sh
cd $PROJECT_ROOT
python analysis/generated_behavior.py --calibrate --score --replot
# classifier robustness: same generations, a deliberately different classifier
python analysis/generated_behavior.py --score --replot --tag robust --no-stem --min-disc 5 --whole-text
echo GENSCORE_DONE
