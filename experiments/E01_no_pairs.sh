#!/bin/bash
# E1 (Table 1): no conditioning channel, and the paired pools D_+ and D_- are left out of the mixture.
# Sign map (0, 0, 0, 0); mixture (1-w_k) D_G + (w_k - 2 lambda) D_S, renormalised. Comparison: E1 -> E2 isolates the
# effect of adding the paired pools. See README, "Known deviations": the Table 1 run of E1 predates the --no-pairs fix.
# Usage: bash experiments/E01_no_pairs.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E1 E01_no_pairs off "NO_PAIRS=1" ""
