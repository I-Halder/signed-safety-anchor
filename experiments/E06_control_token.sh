#!/bin/bash
# E6 (Table 1, control): no anchor; the condition is a trainable prefix token t+ / t- on the labelled pools only
# (Korbak et al., 2023). The label-token embedding rows are trainable (not zeroed); t+ is supplied at evaluation.
# Usage: bash experiments/E06_control_token.sh   (see experiments/_common.sh for the cluster settings it reads)
source "$(dirname "$0")/_common.sh"
submit_experiment E6 E06_control_token off "TOKEN_COND=1" "--token-conditioning"
