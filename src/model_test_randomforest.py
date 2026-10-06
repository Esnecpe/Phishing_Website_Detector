import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

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
# 9. SUMMARY
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