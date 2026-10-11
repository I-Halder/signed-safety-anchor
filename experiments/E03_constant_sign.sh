#!/bin/bash
# E3 (Table 1): anchor with s = +1 on every document, so the sign carries no label.
# Sign map (+1, +1, +1, +1); the functional a is written by the labelled blocks D_+ and D_-.
# Usage: bash experiments/E03_constant_sign.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E3 E03_constant_sign unsigned "" ""
