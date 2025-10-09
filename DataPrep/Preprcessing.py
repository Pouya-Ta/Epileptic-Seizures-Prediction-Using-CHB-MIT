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

# -----------------------------------------------------------------------------
# -------------------------- HELPERS: JSON logging -----------------------------
# -----------------------------------------------------------------------------
# cache: patient_dir -> parsed summary dict
_SUMMARY_CACHE: dict[Path, dict] = {}


def find_and_parse_patient_summary(patient_dir: Path) -> dict:
    """
    Look for a per-patient summary file inside `patient_dir`.
    Typical name: chb01-summary.txt, but we also accept any *summary*.txt.
    Returns an empty dict if nothing is found.
    Results are cached per patient folder.
    """

    if patient_dir in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[patient_dir]

    # Common exact name first (e.g., chb01-summary.txt)
    exact = patient_dir / f"{patient_dir.name}-summary.txt"
    if exact.exists():
        parsed = parse_chb_summary(exact)
        _SUMMARY_CACHE[patient_dir] = parsed
        return parsed

    # Fallback: any *summary*.txt in the folder
    candidates = sorted(patient_dir.glob("*summary*.txt"))
    if candidates:
        parsed = parse_chb_summary(candidates[0])
        _SUMMARY_CACHE[patient_dir] = parsed
        return parsed

    # Nothing found
    _SUMMARY_CACHE[patient_dir] = {}
    return {}

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
