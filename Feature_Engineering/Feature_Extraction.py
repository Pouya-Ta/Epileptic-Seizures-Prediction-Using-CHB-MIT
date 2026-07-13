import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import spectrogram, welch
from scipy.stats import entropy, kurtosis, skew


REPO_ROOT = Path(__file__).resolve().parents[1]

# Place the raw CHB-MIT dataset under Data/CHB-MIT, or point this variable to
# another local copy outside the repository.
RAW_DATA_FOLDER = REPO_ROOT / "Data" / "CHB-MIT"
PREPROCESSED_FOLDER = REPO_ROOT / "Outputs" / "After_preprocessing" / "Preprocessing"
OUTPUT_FOLDER = REPO_ROOT / "Outputs" / "After_feature_extraction"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 256

# EEG frequency bands
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}


def extract_channel_names(summary_path):
    """Extract channel names from a CHB-MIT summary text file."""
    channel_names = []
    capture = False

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("Channels in EDF Files"):
                capture = True
            if capture and line.startswith("Channel"):
                parts = line.split(":")
                if len(parts) > 1:
                    channel_names.append(parts[1].strip())
            if capture and line.startswith("File Name:"):
                break

    return channel_names


def find_summary_file(patient, patient_preprocessed_path):
    """Look for a patient summary file near the outputs, then in raw data."""
    candidate_dirs = [
        Path(patient_preprocessed_path),
        RAW_DATA_FOLDER / patient,
    ]

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        summary_files = [
            file_name
            for file_name in os.listdir(candidate_dir)
            if file_name.endswith("-summary.txt")
        ]
        if summary_files:
            return candidate_dir / summary_files[0]

    return None


def hjorth_parameters(signal):
    activity = np.var(signal)
    mobility = np.sqrt(np.var(np.diff(signal)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal)))
    return activity, mobility, complexity


def zero_crossing_rate(signal):
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def compute_frequency_features(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    median_freq = freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]]
    peak_freq = freqs[np.argmax(psd)]
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    band_powers = {
        band: np.sum(psd[(freqs >= low) & (freqs <= high)]) / np.sum(psd)
        for band, (low, high) in bands.items()
    }
    return mean_freq, median_freq, peak_freq, spec_entropy, band_powers


def compute_stft_features(signal, fs):
    freqs, _, spectrogram_values = spectrogram(signal, fs=fs, nperseg=256)
    spectrogram_norm = spectrogram_values / np.sum(spectrogram_values)
    spec_entropy = np.mean(entropy(spectrogram_norm, axis=0))
    mean_freq = np.mean(
        np.sum(freqs[:, None] * spectrogram_values, axis=0)
        / np.sum(spectrogram_values, axis=0)
    )
    median_freq = np.mean(
        freqs[
            np.argmax(
                np.cumsum(spectrogram_values, axis=0)
                >= np.sum(spectrogram_values, axis=0) / 2,
                axis=0,
            )
        ]
    )
    peak_freq = np.mean(freqs[np.argmax(spectrogram_values, axis=0)])
    return mean_freq, median_freq, peak_freq, spec_entropy


def compute_wavelet_energy(signal, wavelet="db4", level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return [np.sum(np.square(coeff)) for coeff in coeffs[1:]]


feature_types = (
    [
        "Mean",
        "Variance",
        "STD",
        "RMS",
        "Skewness",
        "Kurtosis",
        "Peak-to-Peak",
        "ZCR",
        "Activity",
        "Mobility",
        "Complexity",
        "Mean Frequency",
        "Median Frequency",
        "Peak Frequency",
        "Spectral Entropy",
    ]
    + list(bands.keys())
    + [
        "Mean Frequency (T-F)",
        "Median Frequency (T-F)",
        "Peak Frequency (T-F)",
        "Spectral Entropy (T-F)",
    ]
    + [f"Wavelet Energy L{level}" for level in range(1, 5)]
)
features_per_channel = len(feature_types)


# Preserve the source script's patient subset behavior.
for patient in os.listdir(PREPROCESSED_FOLDER)[14:]:
    patient_path = PREPROCESSED_FOLDER / patient
    if patient_path.is_dir():
        save_path = OUTPUT_FOLDER / patient
        save_path.mkdir(parents=True, exist_ok=True)

        summary_path = find_summary_file(patient, patient_path)
        if summary_path is None:
            print(f"Warning: No summary text file found for {patient}. Skipping...")
            continue

        channel_names = extract_channel_names(summary_path)
        channel_names.pop(0)
        print(channel_names)

        npy_files = [file_name for file_name in os.listdir(patient_path) if file_name.endswith(".npy")]
        if not npy_files:
            print(f"No .npy files found in {patient_path}.")
            continue

        max_channels = 0
        for file_name in npy_files:
            data = np.load(patient_path / file_name)
            max_channels = max(max_channels, data.shape[0])

        if len(channel_names) < max_channels:
            for index in range(len(channel_names), max_channels):
                channel_names.append(f"Ch{index + 1}")
        elif len(channel_names) > max_channels:
            channel_names = channel_names[:max_channels]

        patient_features = []

        for file_name in npy_files:
            file_path = patient_path / file_name
            window_data = np.load(file_path)

            row_features = [file_name]
            n_channels = window_data.shape[0]

            for channel_index in range(n_channels):
                signal = window_data[channel_index]

                mean_val = np.mean(signal)
                variance = np.var(signal)
                std_dev = np.std(signal)
                rms = np.sqrt(np.mean(signal**2))
                skewness = skew(signal)
                kurt = kurtosis(signal)
                peak_to_peak = np.ptp(signal)
                zcr = zero_crossing_rate(signal)
                activity, mobility, complexity = hjorth_parameters(signal)

                mean_freq, median_freq, peak_freq, spec_entropy, band_powers = (
                    compute_frequency_features(signal, SAMPLING_RATE)
                )
                mean_tf, median_tf, peak_tf, spec_entropy_tf = compute_stft_features(
                    signal, SAMPLING_RATE
                )
                wavelet_energy = compute_wavelet_energy(signal)

                channel_features = (
                    [
                        mean_val,
                        variance,
                        std_dev,
                        rms,
                        skewness,
                        kurt,
                        peak_to_peak,
                        zcr,
                        activity,
                        mobility,
                        complexity,
                        mean_freq,
                        median_freq,
                        peak_freq,
                        spec_entropy,
                    ]
                    + list(band_powers.values())
                    + [mean_tf, median_tf, peak_tf, spec_entropy_tf]
                    + wavelet_energy
                )
                row_features.extend(channel_features)

            if n_channels < max_channels:
                missing_channels = max_channels - n_channels
                row_features.extend([np.nan] * (missing_channels * features_per_channel))

            patient_features.append(row_features)

        if not patient_features:
            print(f"No feature data extracted for patient {patient}.")
            continue

        columns = ["Window_Name"]
        for channel_name in channel_names:
            for feature in feature_types:
                columns.append(f"{feature}_channel_{channel_name}")

        expected_total_columns = 1 + max_channels * features_per_channel
        if len(columns) != expected_total_columns:
            print("Warning: Column count does not match expected feature count.")

        final_df = pd.DataFrame(patient_features, columns=columns)
        output_file = save_path / f"{patient}_features.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

print("Feature extraction completed successfully!")
