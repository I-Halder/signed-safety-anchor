#!/bin/bash
# E7 (Table 1, control): E4's anchor with gain beta = 0, so the sign has no effect. It keeps the reparameterised keys,
# the functional a written by labelled blocks, the query noise and the frozen query biases. Table 1 reports the
# four-GPU run of this script.
# Usage: bash experiments/E07_zero_gain.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E7 E07_zero_gain fixed "" "--beta0 0"
