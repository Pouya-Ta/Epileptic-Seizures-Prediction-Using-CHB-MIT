import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="Scaling factor is not defined")
random.seed(42)


REPO_ROOT = Path(__file__).resolve().parents[1]

# Place the raw CHB-MIT dataset under Data/CHB-MIT, or point this variable to
# another local copy outside the repository.
data_folder = str(REPO_ROOT / "Data" / "CHB-MIT")
preprocessed_folder = str(
    REPO_ROOT / "Outputs" / "After_preprocessing" / "Preprocessing"
)
plots_folder = str(REPO_ROOT / "Outputs" / "After_preprocessing" / "Plots")
seizure_summary_csv_path = str(Path(data_folder) / "seizure_summary.csv")

os.makedirs(preprocessed_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)


def band_filter(data, low_freq, high_freq, sfreq):
    """Apply a bandpass filter."""
    nyquist = sfreq / 2
    low, high = low_freq / nyquist, high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="bandpass")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq):
    """Apply a 60 Hz notch filter."""
    b, a = iirnotch(60, 30, sfreq)
    return filtfilt(b, a, data, axis=1)


def soft_thresholding(data, threshold):
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)


def preprocess_eeg(raw_data, fs):
    """Apply filtering and STFT-based denoising."""
    filtered_data = band_filter(raw_data, 0.5, 80, fs)
    centered_data = filtered_data - np.mean(filtered_data, axis=1, keepdims=True)
    cleaned_data = notch_filter(centered_data, fs)
    freqs, _, stft_data = signal.stft(cleaned_data, fs=fs, nperseg=2 * fs)

    mad_noise = np.median(np.abs(stft_data), axis=2, keepdims=True)
    threshold = mad_noise * (0.2 + freqs[None, :, None] / 100)
    denoised_stft = soft_thresholding(stft_data, threshold)
    _, eeg_denoised = signal.istft(denoised_stft, fs=fs, nperseg=2 * fs)

    eeg_denoised = eeg_denoised[:, : cleaned_data.shape[1]]
    return eeg_denoised


def create_windows(data, fs, window_size):
    """Split data into fixed-size windows."""
    step = int(window_size * fs)
    return [data[:, i : i + step] for i in range(0, data.shape[1] - step + 1, step)]


def plot_window(window, fs, ch_names, save_path):
    """Plot a single EEG window and save it."""
    time = np.linspace(0, window.shape[1] / fs, window.shape[1])
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(window):
        plt.plot(time, channel, label=ch_names[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.title(os.path.basename(save_path))
    plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 1.0),
        fontsize="small",
        frameon=True,
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def read_summary_csv(csv_path):
    """Read the seizure summary CSV and group seizures by EDF filename."""
    df = pd.read_csv(csv_path)
    seizure_info = {}
    for _, row in df.iterrows():
        file_name = row["File_name"]
        start_time = int(row["Seizure_start"])
        end_time = int(row["Seizure_stop"])
        if file_name in seizure_info:
            seizure_info[file_name].append((start_time, end_time))
        else:
            seizure_info[file_name] = [(start_time, end_time)]
    return seizure_info


def segment_eeg_with_prediction_logic(
    file_name, preprocessed_data, fs, patient_folder, ch_names, seizure_info
):
    total_preictal_windows = 0
    if file_name not in seizure_info or not seizure_info[file_name]:
        print(f"Info: {file_name} has no seizures, skipping preictal segmentation.")
        return total_preictal_windows

    window_size_sec = 10
    patient_preprocessed_folder = os.path.join(preprocessed_folder, patient_folder)
    os.makedirs(patient_preprocessed_folder, exist_ok=True)

    for seizure_start, seizure_end in seizure_info[file_name]:
        # Preictal period: 10-30 minutes before seizure onset.
        preictal_start = max(0, seizure_start - 900)
        preictal_end = max(0, seizure_start - 300)
        if preictal_end > preictal_start:
            preictal_data = preprocessed_data[
                :, preictal_start * fs : preictal_end * fs
            ]
            preictal_windows = create_windows(preictal_data, fs, window_size_sec)
        else:
            preictal_windows = []
            print(
                f"Warning: For {file_name}, no valid preictal window found "
                "(preictal_end <= preictal_start)."
            )

        for i, window in enumerate(preictal_windows):
            save_path = os.path.join(
                patient_preprocessed_folder, f"{file_name}_preictal_{i}.npy"
            )
            np.save(save_path, window)
            plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))

        total_preictal_windows += len(preictal_windows)
        print(total_preictal_windows)
        print("_" * 162)
        print(f"{file_name} Saved")

    return total_preictal_windows


def segment_interictal_only(
    file_name, preprocessed_data, fs, patient_folder, ch_names, target_count=None
):
    """Create interictal windows, undersampling when a target count is given."""
    window_size_sec = 10
    all_windows = create_windows(preprocessed_data, fs, window_size_sec)

    if target_count is not None and len(all_windows) > target_count:
        windows = random.sample(all_windows, target_count)
    else:
        windows = all_windows

    patient_preprocessed_folder = os.path.join(preprocessed_folder, patient_folder)
    os.makedirs(patient_preprocessed_folder, exist_ok=True)

    for i, window in enumerate(windows):
        save_path = os.path.join(
            patient_preprocessed_folder, f"{file_name}_interictal_{i}.npy"
        )
        np.save(save_path, window)
        plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))

    print("_" * 162)
    print(f"{file_name} Saved")


global_seizure_info = read_summary_csv(seizure_summary_csv_path)


def extract_number(folder_name):
    """Extract the numeric part of a patient folder name for sorting."""
    match = re.search(r"\d+", folder_name)
    return int(match.group()) if match else float("inf")


sorted_folders = sorted(os.listdir(data_folder), key=extract_number)
preictal_count = {}

# Preserve the source script's patient subset behavior.
for patient_folder in sorted_folders[22:]:
    print("_" * 162)
    print("#" * 165)
    print(f"Processing folder: {patient_folder}")
    patient_path = os.path.join(data_folder, patient_folder)

    if not os.path.isdir(patient_path):
        print(f"Skipping {patient_folder}: Not a directory")
        continue

    print("Preictal windows creation started")
    for filename in os.listdir(patient_path):
        print("_" * 162)
        print(f"Preprocessing of file {filename}")
        if not filename.endswith(".edf"):
            continue

        file_path = os.path.join(patient_path, filename)

        if filename in global_seizure_info and global_seizure_info[filename]:
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True)
            except Exception as exc:
                print(f"Error reading {file_path}: {exc}")
                continue

            raw.drop_channels([ch for ch in raw.info["ch_names"] if ch == "-"])
            eeg_data = raw.get_data()
            fs = int(raw.info["sfreq"])
            ch_names = raw.ch_names
            preprocessed_data = preprocess_eeg(eeg_data, fs)
            preictal_count[filename] = []
            preictal_count_one = segment_eeg_with_prediction_logic(
                filename,
                preprocessed_data,
                fs,
                patient_folder,
                ch_names,
                global_seizure_info,
            )
            preictal_count[filename].append(preictal_count_one)
        else:
            print("Skip, no seizure found")

    print("#" * 165)
    print("Interictal windows creation started")

    for filename in os.listdir(patient_path):
        if not filename.endswith(".edf"):
            continue

        file_path = os.path.join(patient_path, filename)

        print("#" * 165)
        print(preictal_count)
        print("#" * 165)

        if filename not in global_seizure_info or not global_seizure_info[filename]:
            try:
                first_key = list(preictal_count.keys())[0]
                interictal_count = preictal_count[first_key]
                if interictal_count[0] > 0:
                    try:
                        raw = mne.io.read_raw_edf(file_path, preload=True)
                    except Exception as exc:
                        print(f"Error reading {file_path}: {exc}")
                        continue

                    raw.drop_channels([ch for ch in raw.info["ch_names"] if ch == "-"])
                    eeg_data = raw.get_data()
                    fs = int(raw.info["sfreq"])
                    ch_names = raw.ch_names
                    preprocessed_data = preprocess_eeg(eeg_data, fs)
                    segment_interictal_only(
                        filename,
                        preprocessed_data,
                        fs,
                        patient_folder,
                        ch_names,
                        target_count=interictal_count[0],
                    )
                    del preictal_count[first_key]
                    interictal_count.pop(0)
                elif interictal_count[0] == 0:
                    print(
                        "Zero number of windows were extracted from previous "
                        "preictal phase"
                    )
                    del preictal_count[first_key]
                    interictal_count.pop(0)
            except IndexError:
                print("Skip the file, already made the interictal file")
        else:
            print("Skip, no more interictal needed or seizure found")


print("Preprocessing, segmentation, and plotting complete!")


# ============================================================================
# Legacy reference from the previous repository version
# ----------------------------------------------------------------------------
# The earlier repo file implemented a different topographic preprocessing
# workflow. The summary TXT parsing and preictal-window helper logic are kept
# below as reference because they may still be useful for future work, but they
# are not part of the active preprocessing script above.

_LEGACY_SUMMARY_CACHE: Dict[Path, Dict[str, Dict[str, Any]]] = {}


def legacy_find_and_parse_patient_summary(patient_dir: Path):
    """Load and cache a per-patient CHB-MIT summary TXT file if one exists."""
    if patient_dir in _LEGACY_SUMMARY_CACHE:
        return _LEGACY_SUMMARY_CACHE[patient_dir]

    exact = patient_dir / f"{patient_dir.name}-summary.txt"
    if exact.exists():
        parsed = legacy_parse_chb_summary(exact)
        _LEGACY_SUMMARY_CACHE[patient_dir] = parsed
        return parsed

    candidates = sorted(patient_dir.glob("*summary*.txt"))
    if candidates:
        parsed = legacy_parse_chb_summary(candidates[0])
        _LEGACY_SUMMARY_CACHE[patient_dir] = parsed
        return parsed

    _LEGACY_SUMMARY_CACHE[patient_dir] = {}
    return {}


def legacy_parse_chb_summary(txt_path) -> Dict[str, Dict[str, Any]]:
    """Parse seizure intervals from the original CHB-MIT summary TXT format."""
    out: Dict[str, Dict[str, Any]] = {}
    cur = None
    cur_file = None
    pending_start = None

    r_name = re.compile(r"^File Name:\s*(\S+)\s*$")
    r_fstart = re.compile(r"^File Start Time:\s*([0-9:]+)\s*$")
    r_fend = re.compile(r"^File End Time:\s*([0-9:]+)\s*$")
    r_nsz = re.compile(r"^Number of Seizures in File:\s*(\d+)\s*$")
    r_ss = re.compile(r"^Seizure Start Time:\s*([0-9]+)\s*seconds\s*$")
    r_se = re.compile(r"^Seizure End Time:\s*([0-9]+)\s*seconds\s*$")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()

            match = r_name.match(line)
            if match:
                if cur_file and cur:
                    out[cur_file] = cur
                cur_file = match.group(1)
                cur = {
                    "file_start_time": None,
                    "file_end_time": None,
                    "n_seizures": 0,
                    "seizures": [],
                }
                pending_start = None
                continue

            if not cur:
                continue

            if match := r_fstart.match(line):
                cur["file_start_time"] = match.group(1)
                continue
            if match := r_fend.match(line):
                cur["file_end_time"] = match.group(1)
                continue
            if match := r_nsz.match(line):
                cur["n_seizures"] = int(match.group(1))
                continue
            if match := r_ss.match(line):
                pending_start = int(match.group(1))
                continue
            if match := r_se.match(line):
                end_sec = int(match.group(1))
                if pending_start is not None:
                    cur["seizures"].append((pending_start, end_sec))
                    pending_start = None

    if cur_file and cur:
        out[cur_file] = cur

    return out


def legacy_preictal_windows_for_file(
    seizures: List[Tuple[int, int]],
    file_len_sec: int,
    start_offset: int = 40 * 60,
    end_offset: int = 20 * 60,
) -> List[Tuple[int, int]]:
    """Derive merged preictal intervals from seizure start times."""
    windows = []
    for seizure_start, _ in seizures:
        start = max(0, seizure_start - start_offset)
        end = max(0, seizure_start - end_offset)
        if end > start:
            windows.append((start, end))

    windows.sort()
    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [
        (max(0, int(start)), min(file_len_sec, int(end)))
        for start, end in merged
        if int(end) > 0
    ]
