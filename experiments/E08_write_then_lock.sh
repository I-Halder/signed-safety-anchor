#!/bin/bash
# E8 (Table 1, scheduling): E4 with the anchor learning-rate multiplier kappa_k = 2 for phases 1-18 and 0 afterwards
# (write early, then lock); sum_k kappa_k = 36 as in E4.
# Usage: bash experiments/E08_write_then_lock.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E8 E08_write_then_lock fixed "" "--kappa-schedule bang --num-phases 36"
