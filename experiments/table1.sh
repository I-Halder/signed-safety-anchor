#!/bin/bash
# Table 1 (body) from the outputs of figure2.sh and figure3.sh -> results/core_table.tex. CPU only, seconds.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/ssa/slurm/common.sh"
cd "$REPO"
python analysis/core_table.py
echo "Table 1 body: $REPO/results/core_table.tex"
