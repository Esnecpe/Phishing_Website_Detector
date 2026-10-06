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
    / "random_forest_model.pkl"
)


# ============================================================
# 2. DATASET SETTINGS
# ============================================================

TEXT_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title",
]

TARGET_COLUMN = "label"

# Keep this True because this is the final configuration
# we validated and tested.
REMOVE_SIMILARITY_INDEX = True


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n===== LOADING DATASET =====")

df = pd.read_csv(DATASET_PATH)

print("Original rows:", df.shape[0])
print("Original columns:", df.shape[1])


# ============================================================
# 4. CLEAN DATASET
# ============================================================

print("\n===== CLEANING DATASET =====")

original_rows = df.shape[0]

# Remove duplicate URLs
df = df.drop_duplicates(
    subset=["URL"]
)

# Remove rows containing missing values
df = df.dropna()

cleaned_rows = df.shape[0]

print("Rows after cleaning:", cleaned_rows)
print("Rows removed:", original_rows - cleaned_rows)


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

columns_to_drop = (
    TEXT_COLUMNS
    + [TARGET_COLUMN]
)

if REMOVE_SIMILARITY_INDEX:
    columns_to_drop.append(
        "URLSimilarityIndex"
    )


X = df.drop(
    columns=columns_to_drop
)

y = df[TARGET_COLUMN]


print("\n===== FEATURE PREPARATION =====")

print(
    "Number of features:",
    X.shape[1]
)

print(
    "URLSimilarityIndex removed:",
    "YES"
    if REMOVE_SIMILARITY_INDEX
    else "NO"
)


# ============================================================
# 6. TRAIN / TEST SPLIT
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


# ============================================================
# 7. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n===== TRAINING RANDOM FOREST =====")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 10. EVALUATION METRICS
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


print("\n===== FINAL RANDOM FOREST RESULTS =====")

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
# 11. CONFUSION MATRIX
# ============================================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n===== TOP 15 MOST IMPORTANT FEATURES =====")

print(
    feature_importance.head(15)
)


# ============================================================
# 14. CREATE FINAL MODEL PACKAGE
# ============================================================

# IMPORTANT:
# Confirmed project interpretation:
#
# 0 = PHISHING
# 1 = LEGITIMATE
#
# The backend will use this mapping later.

label_mapping = {
    0: "PHISHING",
    1: "LEGITIMATE",
}


# Store useful evaluation information with the model.
evaluation_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm.tolist(),
}


model_package = {
    # Trained ML model
    "model": model,

    # Exact feature names and ordering expected by the model
    "feature_names": list(X.columns),

    # Human-readable prediction labels
    "label_mapping": label_mapping,

    # Whether URLSimilarityIndex was excluded
    "remove_similarity_index": REMOVE_SIMILARITY_INDEX,

    # Model evaluation information
    "evaluation_metrics": evaluation_metrics,
}


# ============================================================
# 15. SAVE MODEL PACKAGE
# ============================================================

print("\n===== SAVING FINAL MODEL PACKAGE =====")

joblib.dump(
    model_package,
    MODEL_PATH
)

print(
    "Model package saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 16. VERIFY SAVED MODEL PACKAGE
# ============================================================

print("\n===== VERIFYING SAVED MODEL PACKAGE =====")

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
    "Loaded model type:",
    type(loaded_model).__name__
)

print(
    "Number of model features:",
    loaded_model.n_features_in_
)

print(
    "Number of saved feature names:",
    len(loaded_features)
)

print(
    "Label mapping:",
    loaded_labels
)


# ============================================================
# 17. SAFETY CHECKS
# ============================================================

assert (
    loaded_model.n_features_in_
    == len(loaded_features)
)

assert (
    len(loaded_features)
    == X.shape[1]
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
    "\nModel package verified successfully."
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n===== FINAL MODEL SUMMARY =====")

print(
    "Model:",
    type(model).__name__
)

print(
    "Trees:",
    model.n_estimators
)

print(
    "Features:",
    len(X.columns)
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
    "URLSimilarityIndex removed:",
    REMOVE_SIMILARITY_INDEX
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
    f"Final Accuracy: {accuracy:.6f}"
)

print(
    f"Final F1 Score: {f1:.6f}"
)