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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def update_json(json_path: Path, key: str, data):
    obj = _load_json(json_path)
    obj[str(key)] = _to_py(data)     # sanitize
    _ensure_dir(json_path.parent)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NpEncoder)

def _to_py(obj):
    """Recursive converter to pure Python types (safety net)."""
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_py(v) for v in obj ]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    return obj

# -----------------------------------------------------------------------------
# -------------------------- SUMMARY (.txt) PARSER -----------------------------
# (Uses the exact robust parser we discussed; minor tweaks only)
# -----------------------------------------------------------------------------
def parse_chb_summary(txt_path: str | Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    cur: Dict[str, Any] | None = None
    cur_file: str | None = None
    r_name   = re.compile(r'^File Name:\s*(\S+)\s*$')
    r_fstart = re.compile(r'^File Start Time:\s*([0-9:]+)\s*$')
    r_fend   = re.compile(r'^File End Time:\s*([0-9:]+)\s*$')
    r_nsz    = re.compile(r'^Number of Seizures in File:\s*(\d+)\s*$')
    r_ss     = re.compile(r'^Seizure Start Time:\s*([0-9]+)\s*seconds\s*$')
    r_se     = re.compile(r'^Seizure End Time:\s*([0-9]+)\s*seconds\s*$')

    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        pending_start = None
        for line in f:
            line = line.strip()
            m = r_name.match(line)
            if m:
                if cur_file and cur:
                    out[cur_file] = cur
                cur_file = m.group(1)
                cur = {"file_start_time": None, "file_end_time": None,
                       "n_seizures": 0, "seizures": []}
                pending_start = None
                continue
            if not cur:
                continue
            if (m := r_fstart.match(line)): cur["file_start_time"] = m.group(1); continue
            if (m := r_fend.match(line)):   cur["file_end_time"]   = m.group(1); continue
            if (m := r_nsz.match(line)):    cur["n_seizures"]      = int(m.group(1)); continue
            if (m := r_ss.match(line)):     pending_start          = int(m.group(1)); continue
            if (m := r_se.match(line)):
                end_sec = int(m.group(1))
                if pending_start is not None:
                    cur["seizures"].append((pending_start, end_sec))
                    pending_start = None
                continue
        if cur_file and cur:
            out[cur_file] = cur
    return out

