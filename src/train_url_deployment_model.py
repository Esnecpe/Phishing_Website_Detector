from pathlib import Path
from urllib.parse import urlparse
import ipaddress
import re

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


# ============================================================
# 3. URL FEATURE EXTRACTION
# ============================================================

def normalize_url(url):
    """
    Ensure the URL can be parsed consistently.
    """

    url = str(url).strip()

    if not url.startswith(
        ("http://", "https://")
    ):
        url = "https://" + url

    return url


def extract_url_features(url):
    """
    Extract the exact same 15 URL-only features
    that will later be used during live deployment.
    """

    url = normalize_url(url)

    parsed = urlparse(url)

    domain = (
        parsed.hostname
        or ""
    )


    # --------------------------------------------------------
    # Basic URL / domain values
    # --------------------------------------------------------

    url_length = len(url)

    domain_length = len(domain)


    # --------------------------------------------------------
    # IsDomainIP
    # --------------------------------------------------------

    try:
        ipaddress.ip_address(
            domain
        )

        is_domain_ip = 1

    except ValueError:
        is_domain_ip = 0


    # --------------------------------------------------------
    # TLDLength
    # --------------------------------------------------------

    domain_parts = domain.split(".")

    if len(domain_parts) > 1:

        tld_length = len(
            domain_parts[-1]
        )

    else:

        tld_length = 0


    # --------------------------------------------------------
    # NoOfSubDomain
    # --------------------------------------------------------

    if len(domain_parts) <= 2:

        subdomain_count = 0

    else:

        subdomain_count = (
            len(domain_parts)
            - 2
        )


    # --------------------------------------------------------
    # Letters
    # --------------------------------------------------------

    letter_count = sum(
        character.isalpha()
        for character in url
    )

    if url_length > 0:

        letter_ratio = (
            letter_count
            / url_length
        )

    else:

        letter_ratio = 0.0


    # --------------------------------------------------------
    # Digits
    # --------------------------------------------------------

    digit_count = sum(
        character.isdigit()
        for character in url
    )

    if url_length > 0:

        digit_ratio = (
            digit_count
            / url_length
        )

    else:

        digit_ratio = 0.0


    # --------------------------------------------------------
    # Special URL characters
    # --------------------------------------------------------

    equals_count = (
        url.count("=")
    )

    question_mark_count = (
        url.count("?")
    )

    ampersand_count = (
        url.count("&")
    )


    special_characters = re.findall(
        r"[^A-Za-z0-9]",
        url
    )

    other_special_count = len(
        special_characters
    )


    if url_length > 0:

        special_char_ratio = (
            other_special_count
            / url_length
        )

    else:

        special_char_ratio = 0.0


    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    is_https = (
        1
        if parsed.scheme == "https"
        else 0
    )


    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    features = {
        "URLLength":
            url_length,

        "DomainLength":
            domain_length,

        "IsDomainIP":
            is_domain_ip,

        "TLDLength":
            tld_length,

        "NoOfSubDomain":
            subdomain_count,

        "NoOfLettersInURL":
            letter_count,

        "LetterRatioInURL":
            letter_ratio,

        "NoOfDegitsInURL":
            digit_count,

        "DegitRatioInURL":
            digit_ratio,

        "NoOfEqualsInURL":
            equals_count,

        "NoOfQMarkInURL":
            question_mark_count,

        "NoOfAmpersandInURL":
            ampersand_count,

        "NoOfOtherSpecialCharsInURL":
            other_special_count,

        "SpacialCharRatioInURL":
            special_char_ratio,

        "IsHTTPS":
            is_https,
    }

    return features


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

original_rows = (
    df.shape[0]
)


# Remove duplicate URLs.
df = df.drop_duplicates(
    subset=["URL"]
)


# Keep only rows needed for URL-only deployment.
df = df[
    [
        "URL",
        TARGET_COLUMN,
    ]
].copy()


# Remove missing values.
df = df.dropna()


cleaned_rows = (
    df.shape[0]
)


print(
    "Rows after cleaning:",
    cleaned_rows
)

print(
    "Rows removed:",
    original_rows
    - cleaned_rows
)


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