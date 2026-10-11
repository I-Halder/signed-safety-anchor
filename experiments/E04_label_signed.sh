#!/bin/bash
# E4 (Table 1): the label-signed anchor, the reference method.
# Sign map (+1, -1, +-1, +-1): +1 on aligned blocks, -1 on misaligned blocks, a fresh random sign on every other
# document; gain beta_k = 2 min(k/3, 1), query noise sigma = 0.05, anchor learning-rate multiplier kappa_k = 1.
# Usage: bash experiments/E04_label_signed.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E4 E04_label_signed fixed "" ""
