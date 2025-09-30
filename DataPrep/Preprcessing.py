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
