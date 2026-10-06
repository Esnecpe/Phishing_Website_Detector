import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_preprocessing import load_data, clean_data, prepare_features


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = load_data()

df = clean_data(df)

X, y = prepare_features(
    df,
    remove_similarity_index=True
)


# ============================================================
# 2. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 3. CREATE 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. DEFINE EVALUATION METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# ============================================================
# 5. RUN CROSS-VALIDATION
# ============================================================

print("\n===== 5-FOLD CROSS-VALIDATION =====")
print("Running Random Forest cross-validation...\n")

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# ============================================================
# 6. DISPLAY EACH FOLD
# ============================================================

for i in range(5):
    print(f"Fold {i + 1}")
    print(f"Accuracy:  {results['test_accuracy'][i]:.6f}")
    print(f"Precision: {results['test_precision'][i]:.6f}")
    print(f"Recall:    {results['test_recall'][i]:.6f}")
    print(f"F1 Score:  {results['test_f1'][i]:.6f}")
    print()


# ============================================================
# 7. DISPLAY AVERAGE RESULTS
# ============================================================

print("===== CROSS-VALIDATION AVERAGES =====")

print(
    f"Mean Accuracy:  "
    f"{np.mean(results['test_accuracy']):.6f}"
)

print(
    f"Mean Precision: "
    f"{np.mean(results['test_precision']):.6f}"
)

print(
    f"Mean Recall:    "
    f"{np.mean(results['test_recall']):.6f}"
)

print(
    f"Mean F1 Score:  "
    f"{np.mean(results['test_f1']):.6f}"
)


# ============================================================
# 8. DISPLAY STANDARD DEVIATION
# ============================================================

print("\n===== STANDARD DEVIATION =====")

print(
    f"Accuracy STD:  "
    f"{np.std(results['test_accuracy']):.6f}"
)

print(
    f"Precision STD: "
    f"{np.std(results['test_precision']):.6f}"
)

print(
    f"Recall STD:    "
    f"{np.std(results['test_recall']):.6f}"
)

print(
    f"F1 Score STD:  "
    f"{np.std(results['test_f1']):.6f}"
)