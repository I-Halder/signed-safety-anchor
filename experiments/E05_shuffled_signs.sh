#!/bin/bash
# E5 (Table 1, control): shuffled signs. The labelled blocks keep their gradients and counts, but their sign is a
# pseudo-random function of the block's first tokens, uninformative about the label. Sign map (sigma(b), sigma(b), +-1, +-1).
# Usage: bash experiments/E05_shuffled_signs.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E5 E05_shuffled_signs fixed "" "--permuted-labels"
