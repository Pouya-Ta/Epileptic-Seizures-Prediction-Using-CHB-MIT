from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

# This folder should contain the per-patient feature CSV outputs produced by
# Feature_Extraction.py.
FEATURE_EXTRACTION_DIR = REPO_ROOT / "Outputs" / "After_feature_extraction"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "After_data_preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output directory created (if not existed):", OUTPUT_DIR)


def assign_label(name):
    """Map window names to the original class labels used by the source script."""
    name = str(name).lower()
    if "preictal" in name:
        return -1
    if "interictal" in name:
        return 1
    if "ictal" in name:
        return 0
    return None


# Get the list of patient feature folders.
patient_folders = sorted(FEATURE_EXTRACTION_DIR.iterdir())
print("Found patient folders:", [folder.name for folder in patient_folders])

train_data = []
test_data = []
patient_data = {}

# Identify the columns shared by all patients using the first CSV of each folder.
common_columns = None
for folder_path in patient_folders:
    if folder_path.is_dir():
        csv_files = [file_path.name for file_path in folder_path.iterdir() if file_path.suffix == ".csv"]
        if csv_files:
            sample_df = pd.read_csv(folder_path / csv_files[0])
            if common_columns is None:
                common_columns = set(sample_df.columns)
            else:
                common_columns.intersection_update(sample_df.columns)
            print(
                f"Processed {folder_path.name}: Found common columns in sample file {csv_files[0]}"
            )

common_columns = list(common_columns)
if "Window_Name" not in common_columns:
    common_columns.append("Window_Name")
print("Final common columns across all patients:", common_columns)

# Build each patient dataset, then store it for leave-one-patient-out splits.
for index, folder_path in enumerate(patient_folders):
    if folder_path.is_dir():
        csv_files = [file_path.name for file_path in folder_path.iterdir() if file_path.suffix == ".csv"]
        print(
            f"Processing patient {index + 1}/{len(patient_folders)}: {folder_path.name}, "
            f"found {len(csv_files)} files"
        )

        for csv_file in csv_files:
            file_path = folder_path / csv_file
            df = pd.read_csv(file_path)
            print(f"Reading file: {csv_file}, shape: {df.shape}")

            df = df[common_columns]

            if "Window_Name" in df.columns:
                print("Unique Window_Name values:", df["Window_Name"].unique())
            else:
                print("Error: 'Window_Name' column not found in the dataframe!")

            df["Label"] = df["Window_Name"].apply(assign_label)
            print("Label distribution:", df["Label"].value_counts())

            df = df.dropna(subset=["Label"])

            ictal_count = (df["Label"] == 0).sum()
            df = df[df["Label"] != 0]
            print(f"Removed {ictal_count} ictal records, new shape: {df.shape}")

            if "Window_Name" in df.columns:
                columns = ["Window_Name"] + [
                    column for column in df.columns if column != "Window_Name"
                ]
                df = df[columns]

            preictal_df = df[df["Label"] == -1]
            interictal_df = df[df["Label"] == 1]

            print(
                f"Before balancing: preictal={len(preictal_df)}, interictal={len(interictal_df)}"
            )

            if len(preictal_df) > 0 and len(interictal_df) > len(preictal_df):
                interictal_sampled_df = interictal_df.sample(
                    n=len(preictal_df), random_state=42
                )
                df = pd.concat(
                    [preictal_df, interictal_sampled_df], ignore_index=True
                )
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                print(
                    "After balancing: "
                    f"preictal={len(preictal_df)}, "
                    f"interictal={len(interictal_sampled_df)}"
                )
            else:
                print(
                    "Skipping balancing due to insufficient preictal or interictal data."
                )

            patient_data[folder_path.name] = df


print("Generating LOOCV splits...")

for test_patient in patient_data:
    print(f"Leaving out {test_patient} as test set.")

    test_df = patient_data[test_patient]
    train_dfs = [df for patient, df in patient_data.items() if patient != test_patient]
    train_df = pd.concat(train_dfs, ignore_index=True)

    train_file_path = OUTPUT_DIR / f"train_patient_{test_patient}.csv"
    test_file_path = OUTPUT_DIR / f"test_patient_{test_patient}.csv"

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print(f"Saved LOOCV split: train -> {train_file_path}, test -> {test_file_path}")
