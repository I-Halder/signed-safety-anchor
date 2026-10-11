#!/bin/bash
# E2 (Table 1): no conditioning channel, same data as every arm below (the same-data control).
# Sign map (0, 0, 0, 0); mixture (1-w_k) D_G + (w_k - 2 lambda) D_S + lambda D_+ + lambda D_-, lambda = 0.005.
# Usage: bash experiments/E02_unconditional.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E2 E02_unconditional off "" ""
