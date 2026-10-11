#!/bin/bash
# Submit all ten experiments of Table 1 (E1-E10). Each submits midtraining -> SFT + evaluation -> generations.
# Usage: bash experiments/run_all_table1.sh      (cluster settings: see experiments/_common.sh)
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
for s in E01_no_pairs E02_unconditional E03_constant_sign E04_label_signed E05_shuffled_signs E06_control_token \
         E07_zero_gain E08_write_then_lock E09_positive_only E10_presume_negative; do
  bash "$D/$s.sh"
done
