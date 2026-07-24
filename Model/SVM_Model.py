from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Outputs" / "Train_Test_modification"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "SVM_results"
OUTPUT_CSV = OUTPUT_DIR / "svm_summary_results.csv"
REPORT_DIR = OUTPUT_DIR / "SVM_Reports"
CONF_MATRIX_DIR = OUTPUT_DIR / "SVM_Plots" / "Confusion_Matrix"
ACC_PLOT_DIR = OUTPUT_DIR / "SVM_Plots" / "Accuracy_Comparison"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CONF_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
ACC_PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Expected train/test CSVs under Outputs/Train_Test_modification. "
        "Run Feature_Engineering/Train_Test_Preparation.py first or update DATA_DIR."
    )

results = []


def plot_svm_decision_boundary_2d(X, y, model, patient_id, kernel, save_path):
    """Plot the 2D PCA decision surface used by the source script."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    prediction_grid = model.predict(np.c_[xx.ravel(), yy.ravel()])
    prediction_grid = prediction_grid.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    cmap_light = ListedColormap(["#FFBBBB", "#BBBBFF"])
    plt.contourf(xx, yy, prediction_grid, cmap=cmap_light, alpha=0.5)

    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        color="red",
        label="Ictal (-1)",
        edgecolor="white",
        s=30,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="blue",
        label="Preictal (1)",
        edgecolor="white",
        s=30,
    )

    if hasattr(model, "support_vectors_"):
        plt.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100,
            facecolors="none",
            edgecolors="black",
            label="Support Vectors",
        )

    plt.title(f"SVM Decision Boundary (2D PCA) - {patient_id} ({kernel})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


for train_path in DATA_DIR.iterdir():
    if train_path.name.startswith("final_train_") and train_path.suffix == ".csv":
        patient_id = train_path.stem.replace("final_train_", "")
        test_path = DATA_DIR / f"final_test_{patient_id}.csv"

        if not test_path.exists():
            print(f"Missing test file for {patient_id}, skipping.")
            continue

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        label_col = df_train.columns[-1]
        name_col = df_train.columns[0]

        df_train = df_train[df_train[label_col] != 0]
        df_test = df_test[df_test[label_col] != 0]

        X_train = df_train.drop(columns=[name_col, label_col])
        y_train = df_train[label_col]
        X_test = df_test.drop(columns=[name_col, label_col])
        y_test = df_test[label_col]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Preserve the original search space, including degree values that only
        # matter if the kernel choice is expanded later.
        param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],
            "kernel": ["rbf"],
            "degree": [3, 4, 5],
        }

        grid_search = GridSearchCV(
            SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        y_pred_test = best_model.predict(X_test_scaled)
        y_pred_train = best_model.predict(X_train_scaled)

        test_acc = accuracy_score(y_test, y_pred_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        precision = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        overfit_gap = train_acc - test_acc
        report = classification_report(
            y_test, y_pred_test, target_names=["Ictal (-1)", "Preictal (1)"]
        )

        results.append(
            {
                "Patient": patient_id,
                "Kernel": "rbf",
                "Accuracy": test_acc,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "Train_Accuracy": train_acc,
                "Overfitting_Gap": overfit_gap,
                "Features_Used": X_train.shape[1],
                "Best_C": grid_search.best_params_["C"],
                "Best_gamma": grid_search.best_params_["gamma"],
                "Best_degree": grid_search.best_params_.get("degree", "N/A"),
            }
        )

        report_file = REPORT_DIR / f"report_{patient_id}_rbf.txt"
        with open(report_file, "w", encoding="utf-8") as handle:
            handle.write(f"Patient: {patient_id}\nKernel: rbf\n")
            handle.write(f"Train Accuracy: {train_acc:.4f}\n")
            handle.write(f"Test Accuracy: {test_acc:.4f}\n")
            handle.write(f"Overfitting Gap: {overfit_gap:.4f}\n")
            handle.write("\nClassification Report:\n")
            handle.write(report)

        cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ictal (-1)", "Preictal (1)"],
            yticklabels=["Ictal (-1)", "Preictal (1)"],
        )
        plt.title(f"Confusion Matrix: {patient_id} (rbf)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_DIR / f"confusion_matrix_{patient_id}_rbf.png")
        plt.close()

        plt.figure()
        plt.bar(["Train", "Test"], [train_acc, test_acc], color=["skyblue", "salmon"])
        plt.ylim(0, 1)
        plt.title(f"Train vs Test Accuracy: {patient_id} (rbf)")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(ACC_PLOT_DIR / f"accuracy_comparison_{patient_id}_rbf.png")
        plt.close()

        print(f"Finished {patient_id} | kernel=rbf | test acc={test_acc:.4f}")

        if X_train.shape[1] >= 2:
            decision_plot_dir = OUTPUT_DIR / "SVM_Plots" / "Decision_Boundary_2D"
            decision_plot_dir.mkdir(parents=True, exist_ok=True)

            pca_2d = PCA(n_components=2)
            X_train_2d = pca_2d.fit_transform(X_train_scaled)

            model_2d = SVC(kernel="rbf")
            model_2d.fit(X_train_2d, y_train)

            save_path = decision_plot_dir / f"decision_boundary_{patient_id}_rbf.png"
            plot_svm_decision_boundary_2d(
                X_train_2d, y_train.values, model_2d, patient_id, "rbf", save_path
            )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["Patient", "Kernel"])
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nAll done. Results saved to:\n- {OUTPUT_CSV}\n- {REPORT_DIR}")
