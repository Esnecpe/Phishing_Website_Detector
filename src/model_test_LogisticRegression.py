from pathlib import Path


import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_preprocessing import preprocess_data

def main():
    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(
        remove_similarity_index=True
    )

    # Create and train Logistic Regression
    # Fit scaling and LR together so test rows never influence the scaling statistics.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
        )),
    ])

    # Train
    print("\n===== TRAINING LOGISTIC REGRESSION =====")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")


    # 7. Predict
    y_pred = model.predict(X_test)


    # 8. Evaluate
    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    # These default scores use legitimate class 1; phishing scores are printed below.
    precision = precision_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
    )


    print("\n===== LOGISTIC REGRESSION RESULTS =====")

    print(
        f"Accuracy:  {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall:    {recall:.6f}"
    )

    print(
        f"F1 Score:  {f1:.6f}"
    )


    # Precision-Recall Curve

    # Calculate probabilities once, then select the column belonging to phishing label 0.
    phishing_column = list(model.classes_).index(0)
    probabilities = model.predict_proba(X_test)
    phishing_probability = probabilities[:, phishing_column]

    # Keep the values for the curve; ignore the unused threshold list.
    precisions, recalls, _ = precision_recall_curve(
        y_test,
        phishing_probability,
        pos_label=0,
    )

    #Classification Report

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            y_pred,
        )
    )


    #Standardized Feature Coefficients

    # Pair input names with LR weights; positive weights favor legitimate class 1.
    coefficients = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Coefficient": model.named_steps["classifier"].coef_[0],
        }
    )

    # Sort by magnitude while keeping signed weights for the graph.
    coefficients["Absolute coefficient"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values(
        by="Absolute coefficient",
        ascending=False,
    )

    print("\n===== TOP 15 STANDARDIZED COEFFICIENTS =====")
    print(coefficients.head(15))

    # Phishing-Class Metrics and ROC-AUC

    phish_precision = precision_score(y_test, y_pred, pos_label=0)
    phish_recall = recall_score(y_test, y_pred, pos_label=0)
    phish_f1 = f1_score(y_test, y_pred, pos_label=0)

    # Probability that each test URL is legitimate (class 1)
    legitimate_column = list(model.classes_).index(1)
    y_proba = probabilities[:, legitimate_column]
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n===== PHISHING CLASS (pos_label=0) =====")
    print(f"Phishing Precision: {phish_precision:.6f}")
    print(f"Phishing Recall:    {phish_recall:.6f}")
    print(f"Phishing F1 Score:  {phish_f1:.6f}")

    print("\n===== ROC-AUC =====")
    print(f"ROC-AUC: {roc_auc:.6f}")


    # 5 - Fold Cross-Validation (Training set)


    print("\n===== 5-FOLD CROSS-VALIDATION =====")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Each fold fits a fresh pipeline, including its scaler, using only training folds.
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "phishing_f1": make_scorer(f1_score, pos_label=0),
        },
        error_score="raise",
    )

    for metric in ["accuracy", "precision", "recall", "f1", "phishing_f1"]:
        # Here test means the validation fold, not the reserved 20% test set.
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.capitalize():<10} mean = {scores.mean():.6f}   std = {scores.std():.6f}")



    # FIGURES (saved to figures/ for the report)


    FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
    FIGURES_DIR.mkdir(exist_ok=True)

    # --- Figure 1: Precision-recall curve (phishing is the positive class) ---
    # W05 p28 plots both scores against thresholds. Here we plot precision
    # against recall directly to show the same trade-off in one curve.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="#2a6fb0")
    ax.set_xlabel("Recall: fraction of phishing URLs detected")
    ax.set_ylabel("Precision: fraction of flagged URLs that are phishing")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Logistic Regression - Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lr_precision_recall_curve.png", dpi=200)
    plt.close(fig)

    # --- Figure 2: Top 15 standardized coefficients ---
    top15 = coefficients.head(15).iloc[::-1]  # largest magnitude at top
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["Feature"], top15["Coefficient"], color="#2a6fb0")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Standardized coefficient (positive = legitimate)")
    ax.set_title("Logistic Regression - Top 15 Coefficients")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lr_coefficients.png", dpi=200)
    plt.close(fig)

    # --- Figure 3: ROC curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
        name="Logistic Regression",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_title("Logistic Regression - ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lr_roc_curve.png", dpi=200)
    plt.close(fig)

    print("\n===== FIGURES SAVED =====")
    for name in ["lr_precision_recall_curve.png", "lr_coefficients.png", "lr_roc_curve.png"]:
        print(FIGURES_DIR / name)


    # ============================================================
    # 12. SUMMARY
    # ============================================================

    print("\n===== MODEL SUMMARY =====")

    print(
        "Model: Logistic Regression Classifier"
    )

    print(
        "Regularization C:",
        model.named_steps["classifier"].C,
    )

    print(
        "Training samples:",
        X_train.shape[0],
    )

    print(
        "Testing samples:",
        X_test.shape[0],
    )

    print(
        "Number of features:",
        X_train.shape[1],
    )

    print(
        "URLSimilarityIndex removed: YES"
    )


if __name__ == "__main__":
    main()
