from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from phishing_predictor import PhishingPredictor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

PHISHING_SAMPLES = 5
LEGITIMATE_SAMPLES = 5

RANDOM_STATE = 42


# ============================================================
# 3. LOAD AND CLEAN DATASET
# ============================================================

print("\n===== LOADING DATASET =====")

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Original rows:",
    len(df)
)


# Remove duplicate URLs just like training.
df = df.drop_duplicates(
    subset=["URL"]
)

# Keep only needed columns.
df = df[
    [
        "URL",
        "label",
    ]
].copy()

df = df.dropna()

df = df.reset_index(
    drop=True
)


print(
    "Cleaned rows:",
    len(df)
)


# ============================================================
# 4. RECREATE SAME TRAIN / TEST SPLIT
# ============================================================

print(
    "\n===== RECREATING TEST SET ====="
)


train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)


test_df = df.loc[
    test_indices
].copy()


print(
    "Held-out test URLs:",
    len(test_df)
)


print(
    "\nTest set distribution:"
)

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. SELECT TEST SAMPLES
# ============================================================

phishing_df = test_df[
    test_df["label"] == 0
].sample(
    n=PHISHING_SAMPLES,
    random_state=RANDOM_STATE
)


legitimate_df = test_df[
    test_df["label"] == 1
].sample(
    n=LEGITIMATE_SAMPLES,
    random_state=RANDOM_STATE
)


samples = pd.concat(
    [
        phishing_df,
        legitimate_df,
    ]
)


# Shuffle samples
samples = samples.sample(
    frac=1,
    random_state=RANDOM_STATE
)


# ============================================================
# 6. LOAD PREDICTOR
# ============================================================

print(
    "\n===== LOADING URL MODEL ====="
)

predictor = PhishingPredictor()

print(
    "Model loaded successfully."
)

print(
    "Expected features:",
    len(
        predictor.feature_names
    )
)


# ============================================================
# 7. TEST SAMPLES
# ============================================================

print(
    "\n===== TESTING HELD-OUT URL SAMPLES ====="
)


correct_predictions = 0
total_predictions = 0


for number, (_, row) in enumerate(
    samples.iterrows(),
    start=1
):

    url = row["URL"]

    actual_label = int(
        row["label"]
    )

    actual_name = (
        "PHISHING"
        if actual_label == 0
        else "LEGITIMATE"
    )


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    result = predictor.predict(
        url
    )


    predicted_label = (
        result["prediction"]
    )

    predicted_name = (
        result["label"]
    )


    # --------------------------------------------------------
    # Check correctness
    # --------------------------------------------------------

    correct = (
        predicted_label
        == actual_label
    )


    if correct:
        correct_predictions += 1


    total_predictions += 1


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\n--------------------------------------------"
    )

    print(
        f"Sample {number}"
    )

    print(
        "--------------------------------------------"
    )

    print(
        "URL:"
    )

    print(
        url
    )


    print(
        "\nActual:"
    )

    print(
        actual_name
    )


    print(
        "\nPredicted:"
    )

    print(
        predicted_name
    )


    print(
        "\nRisk Score:"
    )

    print(
        f'{result["risk_score"]:.2f}%'
    )


    print(
        "\nConfidence:"
    )

    print(
        f'{result["confidence"]:.2f}%'
    )


    print(
        "\nCorrect:"
    )

    print(
        "YES"
        if correct
        else "NO"
    )


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

sample_accuracy = (
    correct_predictions
    / total_predictions
) * 100


print(
    "\n============================================"
)

print(
    "            SAMPLE TEST SUMMARY"
)

print(
    "============================================"
)


print(
    "Total URLs tested:",
    total_predictions
)

print(
    "Correct predictions:",
    correct_predictions
)

print(
    "Incorrect predictions:",
    total_predictions
    - correct_predictions
)

print(
    f"Sample accuracy: {sample_accuracy:.2f}%"
)


print(
    "\nExpected label mapping:"
)

print(
    "0 = PHISHING"
)

print(
    "1 = LEGITIMATE"
)


print(
    "\n============================================"
)