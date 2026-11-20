from pathlib import Path
from phishing_predictor import extract_url_features
from data_preprocessing import clean_data, check_numeric_data
import warnings

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "url_random_forest.pkl"
)

LOGISTIC_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "url_logistic_model.pkl"
)

MODEL_PATH.parent.mkdir(exist_ok=True)


# ============================================================
# 2. URL-ONLY FEATURE LIST
# ============================================================

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

TARGET_COLUMN = "label"

def main():
    # ============================================================
    # 4. LOAD DATASET
    # ============================================================

    print(
        "\n===== LOADING DATASET ====="
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        "Original rows:",
        df.shape[0]
    )

    print(
        "Original columns:",
        df.shape[1]
    )


    # ============================================================
    # 5. CLEAN DATASET
    # ============================================================

    print(
        "\n===== CLEANING DATASET ====="
    )

    # Check only URL and label here; the numeric inputs are rebuilt from URL text below.
    df = clean_data(df[["URL", TARGET_COLUMN]])


    # ============================================================
    # 6. REBUILD URL FEATURES
    # ============================================================

    print(
        "\n===== EXTRACTING URL FEATURES ====="
    )

    print(
        "This may take a little time..."
    )


    feature_rows = []


    for url in df["URL"]:

        features = (
            extract_url_features(
                url
            )
        )

        feature_rows.append(
            features
        )


    X = pd.DataFrame(
        feature_rows
    )


    y = (
        df[TARGET_COLUMN]
        .reset_index(drop=True)
    )


    print(
        "Feature extraction complete."
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Feature count:",
        X.shape[1]
    )


    # ============================================================
    # 7. VERIFY FEATURE ORDER
    # ============================================================

    print(
        "\n===== VERIFYING FEATURES ====="
    )

    if list(X.columns) != URL_FEATURES:

        raise ValueError(
            "Extracted URL feature order "
            "does not match URL_FEATURES."
        )


    print(
        "URL feature order verified."
    )


    # ============================================================
    # 8. TARGET DISTRIBUTION
    # ============================================================

    print(
        "\n===== TARGET DISTRIBUTION ====="
    )

    print(
        y.value_counts()
        .sort_index()
    )


    # ============================================================
    # 9. TRAIN / TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


    print(
        "\n===== TRAIN / TEST SPLIT ====="
    )

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    print(
        "Features:",
        X_train.shape[1]
    )

    check_numeric_data(X_train, X_test)


    # ============================================================
    # 10. CREATE RANDOM FOREST
    # ============================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )


    # ============================================================
    # 11. TRAIN MODEL
    # ============================================================

    print(
        "\n===== TRAINING URL-ONLY RANDOM FOREST ====="
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )


    # ============================================================
    # 12. MAKE PREDICTIONS
    # ============================================================

    y_pred = model.predict(
        X_test
    )


    # ============================================================
    # 13. CALCULATE METRICS
    # ============================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )


    print(
        "\n===== URL-ONLY MODEL RESULTS ====="
    )

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


    # ============================================================
    # 14. CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print(
        "\n===== CONFUSION MATRIX ====="
    )

    print(
        cm
    )


    # ============================================================
    # 15. CLASSIFICATION REPORT
    # ============================================================

    print(
        "\n===== CLASSIFICATION REPORT ====="
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # ============================================================
    # 16. FEATURE IMPORTANCE
    # ============================================================

    feature_importance = pd.DataFrame(
        {
            "Feature":
                X_train.columns,

            "Importance":
                model.feature_importances_,
        }
    )


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
    )


    print(
        "\n===== URL FEATURE IMPORTANCE ====="
    )

    print(
        feature_importance
    )


    # ============================================================
    # 17. LABEL MAPPING
    # ============================================================

    label_mapping = {
        0: "PHISHING",
        1: "LEGITIMATE",
    }


    # ============================================================
    # 18. EVALUATION METADATA
    # ============================================================

    evaluation_metrics = {
        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "confusion_matrix":
            cm.tolist(),
    }


    # ============================================================
    # 19. CREATE MODEL PACKAGE
    # ============================================================

    model_package = {

        "model":
            model,

        "feature_names":
            URL_FEATURES,

        "label_mapping":
            label_mapping,

        "evaluation_metrics":
            evaluation_metrics,

        "model_type":
            "RandomForestClassifier",

        "deployment_type":
            "URL_ONLY",

        "n_estimators":
            model.n_estimators,

        "feature_count":
            len(URL_FEATURES),
    }


    # ============================================================
    # 20. SAVE MODEL
    # ============================================================

    print(
        "\n===== SAVING URL DEPLOYMENT MODEL ====="
    )

    joblib.dump(
        model_package,
        MODEL_PATH
    )


    print(
        "Saved to:"
    )

    print(
        MODEL_PATH
    )


    # ============================================================
    # 21. VERIFY SAVED MODEL
    # ============================================================

    print(
        "\n===== VERIFYING SAVED MODEL ====="
    )

    loaded_package = joblib.load(
        MODEL_PATH
    )


    loaded_model = (
        loaded_package["model"]
    )

    loaded_features = (
        loaded_package["feature_names"]
    )

    loaded_labels = (
        loaded_package["label_mapping"]
    )


    print(
        "Loaded model:",
        type(loaded_model).__name__
    )

    print(
        "Model expects features:",
        loaded_model.n_features_in_
    )

    print(
        "Saved feature names:",
        len(loaded_features)
    )

    print(
        "Deployment type:",
        loaded_package[
            "deployment_type"
        ]
    )

    print(
        "Label mapping:",
        loaded_labels
    )


    # ============================================================
    # 22. SAFETY CHECKS
    # ============================================================

    assert (
        loaded_model.n_features_in_
        == 15
    )

    assert (
        len(loaded_features)
        == 15
    )

    assert (
        loaded_features
        == URL_FEATURES
    )

    assert (
        loaded_labels[0]
        == "PHISHING"
    )

    assert (
        loaded_labels[1]
        == "LEGITIMATE"
    )


    print(
        "\nURL deployment model "
        "verified successfully."
    )


    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    print(
        "\n===== URL DEPLOYMENT MODEL SUMMARY ====="
    )

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Trees:",
        model.n_estimators
    )

    print(
        "Deployment type:",
        "URL_ONLY"
    )

    print(
        "Features:",
        len(URL_FEATURES)
    )

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    print(
        "Label 0:",
        label_mapping[0]
    )

    print(
        "Label 1:",
        label_mapping[1]
    )

    print(
        f"Accuracy: {accuracy:.6f}"
    )

    print(
        f"F1 Score: {f1:.6f}"
    )

    print(
        "\nURL-only deployment model "
        "is ready for testing."
    )


    # ============================================================
    # 24. CREATE LOGISTIC REGRESSION
    # ============================================================

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
        )),
    ])


    # ============================================================
    # 25. TRAIN LOGISTIC REGRESSION
    # ============================================================

    print(
        "\n===== TRAINING URL LOGISTIC REGRESSION ====="
    )

    # Stop instead of saving LR if the optimizer fails to converge.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        logistic_model.fit(
            X_train,
            y_train,
        )

    print(
        "Training complete."
    )


    # ============================================================
    # 26. EVALUATE LOGISTIC REGRESSION
    # ============================================================

    logistic_predictions = logistic_model.predict(X_test)
    phishing_column = list(logistic_model.classes_).index(0)
    logistic_probabilities = logistic_model.predict_proba(X_test)[:, phishing_column]

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions,
    )

    logistic_precision = precision_score(
        y_test,
        logistic_predictions,
        pos_label=0,
    )

    logistic_recall = recall_score(
        y_test,
        logistic_predictions,
        pos_label=0,
    )

    logistic_f1 = f1_score(
        y_test,
        logistic_predictions,
        pos_label=0,
    )

    # Match the target to the phishing-probability column: True means phishing.
    logistic_auc = roc_auc_score(
        y_test == 0,
        logistic_probabilities,
    )

    print(
        "\n===== URL LOGISTIC REGRESSION RESULTS ====="
    )
    print(f"Accuracy:           {logistic_accuracy:.6f}")
    print(f"Phishing Precision: {logistic_precision:.6f}")
    print(f"Phishing Recall:    {logistic_recall:.6f}")
    print(f"Phishing F1 Score:  {logistic_f1:.6f}")
    print(f"ROC-AUC:            {logistic_auc:.6f}")


    # ============================================================
    # 27. SAVE LOGISTIC REGRESSION FOR THE GUI
    # ============================================================


    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH,
    )

    print(
        "Saved to:"
    )
    print(
        LOGISTIC_MODEL_PATH
    )
    print(
        "\nBoth URL-only models are ready for the GUI."
    )


if __name__ == "__main__":
    main()
