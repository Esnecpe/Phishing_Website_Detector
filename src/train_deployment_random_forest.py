from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. FILE PATHS
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
    / "deployment_random_forest.pkl"
)


# ============================================================
# 2. DEPLOYMENT FEATURES
# ============================================================

# These are the 41 features currently reproducible
# by feature_extractor.py.

DEPLOYMENT_FEATURES = [
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
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

TARGET_COLUMN = "label"


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n===== LOADING DATASET =====")

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
# 4. CLEAN DATASET
# ============================================================

print("\n===== CLEANING DATASET =====")

original_rows = df.shape[0]

# Remove duplicate URLs before splitting.
df = df.drop_duplicates(
    subset=["URL"]
)

# Remove rows with missing values.
df = df.dropna()

cleaned_rows = df.shape[0]

print(
    "Rows after cleaning:",
    cleaned_rows
)

print(
    "Rows removed:",
    original_rows - cleaned_rows
)


# ============================================================
# 5. VERIFY REQUIRED FEATURES
# ============================================================

print("\n===== VERIFYING DEPLOYMENT FEATURES =====")

missing_features = [
    feature
    for feature in DEPLOYMENT_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "The following deployment features are missing "
        f"from the dataset: {missing_features}"
    )

print(
    "All deployment features found."
)

print(
    "Deployment feature count:",
    len(DEPLOYMENT_FEATURES)
)


# ============================================================
# 6. PREPARE X AND y
# ============================================================

X = df[
    DEPLOYMENT_FEATURES
].copy()

y = df[
    TARGET_COLUMN
].copy()


print("\n===== DEPLOYMENT DATA =====")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "\nTarget distribution:"
)

print(
    y.value_counts().sort_index()
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\n===== TRAIN / TEST SPLIT =====")

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


# ============================================================
# 8. CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n===== TRAINING DEPLOYMENT RANDOM FOREST =====")

model.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 11. CALCULATE METRICS
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


print("\n===== DEPLOYMENT MODEL RESULTS =====")

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
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===== CONFUSION MATRIX =====")

print(
    cm
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 14. FEATURE IMPORTANCE
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


print("\n===== TOP 15 DEPLOYMENT FEATURES =====")

print(
    feature_importance.head(15)
)


# ============================================================
# 15. LABEL MAPPING
# ============================================================

# Dataset label interpretation:
#
# 0 = PHISHING
# 1 = LEGITIMATE

label_mapping = {
    0: "PHISHING",
    1: "LEGITIMATE",
}


# ============================================================
# 16. EVALUATION METADATA
# ============================================================

evaluation_metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
}


# ============================================================
# 17. CREATE DEPLOYMENT MODEL PACKAGE
# ============================================================

model_package = {

    # Trained classifier
    "model":
        model,

    # Exact 41 deployment features
    "feature_names":
        DEPLOYMENT_FEATURES,

    # Human-readable labels
    "label_mapping":
        label_mapping,

    # Evaluation results
    "evaluation_metrics":
        evaluation_metrics,

    # Useful metadata
    "model_type":
        "RandomForestClassifier",

    "n_estimators":
        model.n_estimators,

    "deployment_feature_count":
        len(DEPLOYMENT_FEATURES),
}


# ============================================================
# 18. SAVE MODEL PACKAGE
# ============================================================

print("\n===== SAVING DEPLOYMENT MODEL =====")

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
# 19. VERIFY SAVED PACKAGE
# ============================================================

print("\n===== VERIFYING SAVED MODEL =====")

loaded_package = joblib.load(
    MODEL_PATH
)

loaded_model = loaded_package[
    "model"
]

loaded_features = loaded_package[
    "feature_names"
]

loaded_labels = loaded_package[
    "label_mapping"
]


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
    "Label mapping:",
    loaded_labels
)


# ============================================================
# 20. SAFETY CHECKS
# ============================================================

assert (
    loaded_model.n_features_in_
    == 41
)

assert (
    len(loaded_features)
    == 41
)

assert (
    loaded_features
    == DEPLOYMENT_FEATURES
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
    "\nDeployment model verified successfully."
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n===== DEPLOYMENT MODEL SUMMARY =====")

print(
    "Model:",
    type(model).__name__
)

print(
    "Trees:",
    model.n_estimators
)

print(
    "Deployment features:",
    len(DEPLOYMENT_FEATURES)
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
    "\nDeployment model is ready for backend integration."
)