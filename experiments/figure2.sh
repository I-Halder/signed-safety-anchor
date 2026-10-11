#!/bin/bash
#SBATCH --job-name=ssa-figure2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# Figure 2 (fit and preference: operating points and the decomposition of the deployed margin into the two candidate
# likelihoods) and the likelihood statistics behind Table 1 and Appendices B and H. CPU only, a few minutes; do not
# send it to a GPU partition. Run after the experiments have finished: bash experiments/figure2.sh, or submit it with
# sbatch --partition=<cpu partition> experiments/figure2.sh from the repository root.
#   reads   $SSA_ROOT/runs/<run dir>/eval_phase*.json and $SSA_ROOT/runs/<run dir>_sft/eval_post_sft.json
#   writes  results/trajectories.csv, results/scores.json, results/audit.json, results/*_table.tex,
#           figures/conditional_tradeoff_story.{pdf,png} (Figure 2), figures/routing_gap_vs_gain.{pdf,png},
#           figures/likelihood_summary.{pdf,png}, figures/trajectories.{pdf,png}
set -euo pipefail
# Repository root: the submit directory under sbatch (which runs a spooled copy of this file), else this file's parent.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/arms.py" ]; then REPO=$SLURM_SUBMIT_DIR; else REPO=$(cd "$(dirname "$0")/.." && pwd); fi
source "$REPO/ssa/slurm/common.sh"
cd "$REPO"
python analysis/plot_trajectories.py --compute --run-root "$SSA_ROOT/runs"   # per-phase trajectories -> results/trajectories.csv
python analysis/likelihood_analysis.py --collect "$SSA_ROOT/runs"            # per-scenario scores -> results/scores.json; statistics -> results/audit.json
python analysis/tradeoff_figure.py                                           # Figure 2 and the pair-mass table
python analysis/plot_trajectories.py --replot
echo "Figure 2: $REPO/figures/conditional_tradeoff_story.pdf"
