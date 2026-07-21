from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "Outputs" / "After_data_preparation"
SCORE_DIR = REPO_ROOT / "Outputs" / "Feature_scoring"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "After_univariate"
TOP_N = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        "Expected prepared training CSVs under Outputs/After_data_preparation. "
        "Run Feature_Engineering/Data_Preparation.py first or update TRAIN_DIR."
    )


# Process every generated training split and keep the highest-ranked features.
for train_file_path in TRAIN_DIR.iterdir():
    if train_file_path.name.startswith("train_patient_") and train_file_path.suffix == ".csv":
        patient_id = train_file_path.name.replace("train_", "").replace(".csv", "")
        feature_scores_path = SCORE_DIR / f"feature_scores_{patient_id}.csv"
        output_path = OUTPUT_DIR / f"train_{patient_id}_univariate_features.csv"

        print(f"\nProcessing: {train_file_path.name}")
        print(f"Loading feature scores from: {feature_scores_path}")

        feature_scores_df = pd.read_csv(feature_scores_path)
        feature_scores_df = feature_scores_df.sort_values(
            by="Vote_Score", ascending=False
        )
        selected_features = feature_scores_df.head(TOP_N)["Feature"].tolist()

        print(f"Top {TOP_N} features selected.")

        df = pd.read_csv(train_file_path)

        # Keep only features that are still present in the prepared training file.
        selected_features = [feature for feature in selected_features if feature in df.columns]
        print(f"Final selected features after validation: {len(selected_features)}")

        # Preserve the source script's label-last and ID-first column assumptions.
        final_df = df[selected_features + [df.columns[-1], df.columns[0]]]

        if "Window_Name" in final_df.columns:
            columns = [
                "Window_Name",
                *[column for column in final_df.columns if column != "Window_Name"],
            ]
            final_df = final_df[columns]

        final_df.to_csv(output_path, index=False)
        print(f"Saved selected features to: {output_path}")
