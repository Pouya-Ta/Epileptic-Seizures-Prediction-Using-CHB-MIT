from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "Outputs" / "After_univariate"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "After_model_based"
TOP_K = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if not INPUT_DIR.exists():
    raise FileNotFoundError(
        "Expected univariate-selection CSVs under Outputs/After_univariate. "
        "Run Feature_Engineering/Univariate_Selection.py first or update INPUT_DIR."
    )


# Train a RandomForest on each prepared training file and keep the top-ranked features.
for input_path in INPUT_DIR.iterdir():
    if input_path.name.endswith("_univariate_features.csv"):
        output_filename = input_path.name.replace(
            "_univariate_features.csv", "_model_based.csv"
        )
        output_path = OUTPUT_DIR / output_filename

        df = pd.read_csv(input_path)
        name_col = df.columns[0]
        label_col = df.columns[-1]

        X = df.drop(columns=[name_col, label_col])
        y = df[label_col]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        feature_importances = list(zip(X.columns, model.feature_importances_))
        feature_importances.sort(key=lambda item: item[1], reverse=True)

        top_features = [feature for feature, _ in feature_importances[:TOP_K]]
        reduced_df = df[[name_col] + top_features + [label_col]]

        # Keep Window_Name first when it is present in the selected output.
        if "Window_Name" in reduced_df.columns:
            columns = [
                "Window_Name",
                *[column for column in reduced_df.columns if column != "Window_Name"],
            ]
            reduced_df = reduced_df[columns]

        reduced_df.to_csv(output_path, index=False)
        print(f"{input_path.name}: top {TOP_K} features saved to {output_filename}")
