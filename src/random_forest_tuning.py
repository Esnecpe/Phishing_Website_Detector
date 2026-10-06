import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
# 1. LOAD PREPROCESSED DATA
# ============================================================

X_train, X_test, y_train, y_test = preprocess_data(
    remove_similarity_index=True
)


# ============================================================
# 2. CREATE BASE RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 3. DEFINE HYPERPARAMETER SEARCH SPACE
# ============================================================

parameter_space = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


# ============================================================
# 4. RANDOMIZED SEARCH
# ============================================================

random_search = RandomizedSearchCV(
    estimator=rf_model,

    param_distributions=parameter_space,

    # Only test a limited number of combinations.
    # This keeps the search manageable on the large dataset.
    n_iter=10,

    # 3-fold CV for each parameter combination
    cv=3,

    # Optimize primarily for F1 score
    scoring="f1",

    # Use all available CPU cores
    n_jobs=-1,

    # Display progress
    verbose=2,

    random_state=42,

    # Keep training scores for comparison
    return_train_score=True,
)


# ============================================================
# 5. RUN HYPERPARAMETER TUNING
# ============================================================

print("\n===== RANDOM FOREST HYPERPARAMETER TUNING =====")
print("Starting RandomizedSearchCV...")
print("This may take some time because the dataset is large.\n")

random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 6. BEST PARAMETERS
# ============================================================

print("\n===== BEST PARAMETERS =====")

print(
    random_search.best_params_
)


print("\n===== BEST CROSS-VALIDATION SCORE =====")

print(
    f"Best F1 Score: "
    f"{random_search.best_score_:.6f}"
)


# ============================================================
# 7. GET BEST MODEL
# ============================================================

best_model = random_search.best_estimator_


# ============================================================
# 8. TEST BEST MODEL
# ============================================================

y_pred = best_model.predict(
    X_test
)


# ============================================================
# 9. FINAL TEST RESULTS
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


print("\n===== TUNED RANDOM FOREST RESULTS =====")

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
# 10. CONFUSION MATRIX
# ============================================================

print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_,
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