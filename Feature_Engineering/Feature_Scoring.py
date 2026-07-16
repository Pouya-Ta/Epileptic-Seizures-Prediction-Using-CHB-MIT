from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "Outputs" / "After_data_preparation"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "Feature_scoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fisher_score(features, labels):
    """Compute the Fisher score for each feature column."""
    unique_classes = np.unique(labels)
    num_features = features.shape[1]
    overall_mean = np.mean(features, axis=0)

    numerator = np.zeros(num_features)
    denominator = np.zeros(num_features)

    for current_class in unique_classes:
        class_mask = labels == current_class
        class_mean = np.mean(features[class_mask], axis=0)
        class_variance = np.var(features[class_mask], axis=0)
        class_count = np.sum(class_mask)

        numerator += class_count * (class_mean - overall_mean) ** 2
        denominator += class_count * class_variance

    return numerator / (denominator + 1e-10)


print(f"Scanning input directory for train_*.csv files in: {INPUT_DIR}")

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        "Expected data-preparation outputs under Outputs/After_data_preparation. "
        "Run Feature_Engineering/Data_Preparation.py first or point INPUT_DIR to the generated CSVs."
    )

print([path.name for path in INPUT_DIR.iterdir()])

for file_path in INPUT_DIR.iterdir():
    if file_path.name.startswith("train_") and file_path.suffix == ".csv":
        patient_id = file_path.stem.replace("train_", "")
        output_path = OUTPUT_DIR / f"feature_scores_{patient_id}.csv"

        print(f"\nProcessing {file_path.name}...")

        df = pd.read_csv(file_path)
        print(f"Loaded shape: {df.shape}")

        # The source script uses the first column as an identifier and the last as the label.
        feature_columns = df.columns[1:-1]
        labels = df.iloc[:, -1]

        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        df[feature_columns] = df[feature_columns].fillna(0)

        if df[feature_columns].isnull().any().any():
            print(f"Warning: NaNs still present in {file_path.name}")
            continue

        f_values, _ = f_classif(df[feature_columns], labels)
        fisher_values = fisher_score(df[feature_columns].values, labels.values)
        mi_values = mutual_info_classif(
            df[feature_columns], labels, discrete_features="auto"
        )

        num_features = len(feature_columns)
        score_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "f_classif": f_values,
                "fisher": fisher_values,
                "mutual_info": mi_values,
            }
        )

        for method in ["f_classif", "fisher", "mutual_info"]:
            score_df[f"{method}_rank"] = score_df[method].rank(
                ascending=False, method="min"
            )
            score_df[f"{method}_vote"] = num_features - score_df[f"{method}_rank"]

        score_df["Vote_Score"] = score_df[
            [f"{method}_vote" for method in ["f_classif", "fisher", "mutual_info"]]
        ].sum(axis=1)
        score_df = score_df.sort_values(by="Vote_Score", ascending=False)

        feature_score_df = score_df[["Feature", "Vote_Score"]].copy()
        feature_score_df.to_csv(output_path, index=False)
        print(f"Saved feature scores to: {output_path}")
