from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_preprocessing import preprocess_data


# ============================================================
# 1. GET PREPROCESSED DATA
# ============================================================

X_train, X_test, y_train, y_test = preprocess_data(
    remove_similarity_index=True
)


# ============================================================
# 2. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("\n===== TRAINING RANDOM FOREST =====")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

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


print("\n===== RANDOM FOREST RESULTS =====")

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
# 6. CONFUSION MATRIX
# ============================================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred,
)

print(cm)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
    )
)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False,
)


print("\n===== TOP 15 MOST IMPORTANT FEATURES =====")

print(
    feature_importance.head(15)
)


# ============================================================
# 9. PHISHING-CLASS METRICS AND ROC-AUC
# ============================================================
# Label mapping: 0 = Phishing, 1 = Legitimate.
# The metrics above treat "Legitimate" (1) as the positive class
# (scikit-learn default). For a phishing detector, the phishing
# class matters most, so we also report it with pos_label=0.

phish_precision = precision_score(y_test, y_pred, pos_label=0)
phish_recall = recall_score(y_test, y_pred, pos_label=0)
phish_f1 = f1_score(y_test, y_pred, pos_label=0)

# Probability that each test URL is legitimate (class 1)
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print("\n===== PHISHING CLASS (pos_label=0) =====")
print(f"Phishing Precision: {phish_precision:.6f}")
print(f"Phishing Recall:    {phish_recall:.6f}")
print(f"Phishing F1 Score:  {phish_f1:.6f}")

print("\n===== ROC-AUC =====")
print(f"ROC-AUC: {roc_auc:.6f}")


# ============================================================
# 10. 5-FOLD CROSS-VALIDATION (TRAINING SET ONLY)
# ============================================================
# Checks that the high test score is stable and not the result
# of one lucky train/test split. The test set is not used here.

print("\n===== 5-FOLD CROSS-VALIDATION =====")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"],
)

for metric in ["accuracy", "precision", "recall", "f1"]:
    scores = cv_results[f"test_{metric}"]
    print(f"{metric.capitalize():<10} mean = {scores.mean():.6f}   std = {scores.std():.6f}")


# ============================================================
# 11. FIGURES (saved to figures/ for the report)
# ============================================================

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# --- Figure 1: Confusion matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Phishing (0)", "Legitimate (1)"],
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Random Forest - Confusion Matrix")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "rf_confusion_matrix.png", dpi=200)
plt.close(fig)

# --- Figure 2: Top 15 feature importances ---
top15 = feature_importance.head(15).iloc[::-1]  # reverse so largest is on top
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top15["Feature"], top15["Importance"], color="#2a6fb0")
ax.set_xlabel("Importance (mean decrease in impurity)")
ax.set_title("Random Forest - Top 15 Feature Importances")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "rf_feature_importance.png", dpi=200)
plt.close(fig)

# --- Figure 3: ROC curve ---
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    y_test,
    y_proba,
    name="Random Forest",
    ax=ax,
)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_title("Random Forest - ROC Curve")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "rf_roc_curve.png", dpi=200)
plt.close(fig)

print("\n===== FIGURES SAVED =====")
for name in ["rf_confusion_matrix.png", "rf_feature_importance.png", "rf_roc_curve.png"]:
    print(FIGURES_DIR / name)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n===== MODEL SUMMARY =====")

print(
    "Model: Random Forest Classifier"
)

print(
    "Number of trees:",
    model.n_estimators,
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