#!/bin/bash
# E9 (Table 1, label regime): only aligned blocks are labelled (+1); misaligned blocks are treated as unlabelled and
# get a random sign. Sign map (+1, +-1, +-1, +-1); only D_+ writes the functional a.
# Usage: bash experiments/E09_positive_only.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E9 E09_positive_only fixed "" "--positive-only"
