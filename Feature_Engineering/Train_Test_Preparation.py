from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "Outputs" / "After_model_based"
# This folder should contain the leave-one-patient-out test splits from Data_Preparation.py.
TEST_DIR = REPO_ROOT / "Outputs" / "After_data_preparation"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "Train_Test_modification"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_and_standardize(data):
    """Apply the same feature transformation steps used in the source script."""
    data_log = np.log1p(data)

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_log)

    return standardized_data


if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        "Expected model-based training CSVs under Outputs/After_model_based. "
        "Run Feature_Engineering/Model_Based_Selection.py first or update TRAIN_DIR."
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        "Expected data-preparation CSVs under Outputs/After_data_preparation. "
        "Run Feature_Engineering/Data_Preparation.py first or update TEST_DIR."
    )


# Match each train split with its corresponding held-out patient test split.
for train_path in TRAIN_DIR.iterdir():
    if train_path.name.endswith("_model_based.csv"):
        patient_id = train_path.name.replace("train_patient_", "").replace(
            "_model_based.csv", ""
        )
        test_path = TEST_DIR / f"test_patient_{patient_id}.csv"

        if not test_path.exists():
            print(f"Test file for {patient_id} not found, skipping.")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        selected_columns = train_df.columns
        test_df = test_df[selected_columns]

        train_df = train_df.interpolate(method="linear")
        test_df = test_df.interpolate(method="linear")

        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_df = shuffle(train_df, random_state=42)
        test_df = shuffle(test_df, random_state=42)

        train_features = train_df.drop(columns=["Window_Name", "Label"])
        test_features = test_df.drop(columns=["Window_Name", "Label"])
        train_standardized_features = log_and_standardize(train_features)
        test_standardized_features = log_and_standardize(test_features)

        # Preserve the source script's label extraction even though the arrays are not reused later.
        train_cleaned_labels = train_df["Label"].values
        test_cleaned_labels = test_df["Label"].values

        train_df = train_df[
            ["Window_Name"]
            + [
                column
                for column in train_df.columns
                if column not in ["Window_Name", "Label"]
            ]
            + ["Label"]
        ]
        test_df = test_df[
            ["Window_Name"]
            + [
                column for column in test_df.columns if column not in ["Window_Name", "Label"]
            ]
            + ["Label"]
        ]

        train_output_path = OUTPUT_DIR / f"final_train_{patient_id}.csv"
        test_output_path = OUTPUT_DIR / f"final_test_{patient_id}.csv"

        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        print(f"Saved final train/test files for {patient_id}")
