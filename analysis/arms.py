"""The experiment roster: one place that maps experiment IDs to run directories, names and roles.

Every analysis script imports this module, so adding an experiment is a one-line change here.
"""
from __future__ import annotations

from collections import OrderedDict

# id, run directory (under $SSA_ROOT/runs), short name, training sign map (aligned, misaligned, other)
_ROSTER = [
    ("E1",  "E01_no_pairs",         "No pairs",                  ("0", "0", "0")),
    ("E2",  "E02_unconditional",    "Unconditional",             ("0", "0", "0")),
    ("E3",  "E03_constant_sign",    "Constant sign",             ("+1", "+1", "+1")),
    ("E4",  "E04_label_signed",     "Label-signed anchor",       ("+1", "-1", "±1")),
    ("E5",  "E05_shuffled_signs",   "Shuffled signs",            ("σ(b)", "σ(b)", "±1")),
    ("E6",  "E06_control_token",    "Control token",             ("t+", "t-", "none")),
    ("E7",  "E07_zero_gain",        "Zero gain",                 ("0", "0", "0")),
    ("E8",  "E08_write_then_lock",  "Write then lock",           ("+1", "-1", "±1")),
    ("E9",  "E09_positive_only",    "Positive only",             ("+1", "±1", "±1")),
    ("E10", "E10_presume_negative", "Positive vs.\\ unlab.",     ("+1", "-1", "-1")),
    ("E11", "E11_taper",            "Taper (write early)",       ("+1", "-1", "±1")),
    ("E12", "E12_ramp",             "Ramp (write late)",         ("+1", "-1", "±1")),
    ("E13", "E13_flat_matched",     "Uniform, exposure-matched", ("+1", "-1", "±1")),
    ("E14", "E14_late_rectangle",   "Late rectangle",            ("+1", "-1", "±1")),
]

IDS = [r[0] for r in _ROSTER]
RUN_DIR = OrderedDict((r[0], r[1]) for r in _ROSTER)
NAME = OrderedDict((r[0], r[2]) for r in _ROSTER)
SIGN_MAP = OrderedDict((r[0], r[3]) for r in _ROSTER)

# run directory -> experiment id
DIR_TO_ID = {d: i for i, d in RUN_DIR.items()}

# the configurations of Table 1, in row order; the scheduling controls E11-E14 are appendix material
TABLE1 = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
SCHEDULES = ["E14", "E12", "E4", "E11", "E13", "E8"]          # ordered by cumulative nominal LR exposure


def to_id(name: str) -> str:
    """Experiment id from a run-directory name or an experiment id."""
    return DIR_TO_ID.get(name, name)
