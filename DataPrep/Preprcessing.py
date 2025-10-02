import os, re, json, math, random, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, OrderedDict as TOrdered
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA
import json, numpy as np

# -----------------------------------------------------------------------------
# --------------------------- USER CONFIG (edit) -------------------------------
# -----------------------------------------------------------------------------
DATASET_ROOT = r"D:\Project\CHB-MIT\Original_dataset\CHB-MIT"  # your input root
OUT_ROOT     = r"F:\Projects\CHB_MIT_topographic_pr\Processed" # where to write

# canonical 21 bipolar pairs to KEEP (drop any others after dedupe)
CANONICAL_21 = [
 'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
]

# band config for tensors
BANDS = [
    ("delta", (0.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 12.0)),
    ("beta",  (12.0, 30.0)),
    ("gamma", (30.0, 45.0)),
]

TENSOR_RES = 64
NORMALIZE  = "relative"  # "none" | "relative" | "log" | "zscore"

# window policy
WIN_LEN_S    = 2.0       # 2-second windows
WIN_STRIDE_S = 2.0       # non-overlapping
PRE_ICTAL_START_OFFSET = 40*60  # 40 min before onset
PRE_ICTAL_END_OFFSET   = 20*60  # 20 min before onset
RANDOM_BASELINE_LEN    = 20*60  # 20 minutes if no seizure in file

# RNG seed for reproducible random baselines (derived from filename)
RANDOM_SEED = 1337
