#!/bin/bash
#SBATCH --job-name=ssa-figure3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# Figure 3 (generated behaviour after midtraining) and column R of Table 1. CPU only, a few minutes; do not send it to
# a GPU partition. Run after the generation jobs of the experiments have finished: bash experiments/figure3.sh, or
# sbatch --partition=<cpu partition> experiments/figure3.sh from the repository root.
#   reads   $SSA_ROOT/generated_behavior/<experiment id>/generations.rank*.jsonl (written by the experiment scripts)
#   writes  results/generated_behavior/{generations.<id>.jsonl, classifier_calibration.json, generated_behavior.json,
#           generated_behavior_robust.json, *.npz}, figures/generated_behavior.{pdf,png} (Figure 3),
#           figures/generated_behavior_robust.{pdf,png} (the classifier-robustness re-score of Appendix E)
set -euo pipefail
# Repository root: the submit directory under sbatch (which runs a spooled copy of this file), else this file's parent.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/arms.py" ]; then REPO=$SLURM_SUBMIT_DIR; else REPO=$(cd "$(dirname "$0")/.." && pwd); fi
source "$REPO/ssa/slurm/common.sh"
cd "$REPO"
OUT=$REPO/results/generated_behavior; mkdir -p "$OUT"
n=0
for d in "$SSA_ROOT"/generated_behavior/E*/; do                             # one file per experiment, all stages and conditions
  eid=$(basename "$d"); cat "$d"/generations.rank*.jsonl > "$OUT/generations.$eid.jsonl"; n=$((n+1))
done
echo "collected the generations of $n experiments"
export GB_OUT_DIR=$OUT GB_FIG_DIR=$REPO/figures
python analysis/generated_behavior.py --calibrate --score --replot           # chance floor, rates, contrasts, Figure 3
python analysis/generated_behavior.py --score --replot --tag robust --no-stem --min-disc 5 --whole-text
echo "Figure 3: $REPO/figures/generated_behavior.pdf"
