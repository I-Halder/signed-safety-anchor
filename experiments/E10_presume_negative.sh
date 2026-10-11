#!/bin/bash
# E10 (Table 1, label regime): +1 on aligned blocks and -1 on every other document (unlabelled text presumed
# negative); every block writes the functional a. Sign map (+1, -1, -1, -1).
# Usage: bash experiments/E10_presume_negative.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E10 E10_presume_negative fixed "" "--positive-only --unlabelled-negative --plasticity-all"
