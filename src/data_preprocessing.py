from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

# Text columns not used directly in Random Forest
TEXT_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title",
]

# Target column
TARGET_COLUMN = "label"


def load_data():
    """
    Load the phishing dataset.
    """
    df = pd.read_csv(DATASET_PATH)
    return df


def inspect_data(df):
    """
    Display basic information about the original dataset.
    """
    print("\n===== DATASET INFORMATION =====")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n===== DATA TYPES =====")
    print(df.dtypes.value_counts())

    print("\n===== MISSING VALUES =====")
    print("Total missing values:", df.isnull().sum().sum())

    print("\n===== DUPLICATE ROWS =====")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n===== DUPLICATE URLs =====")
    print("Duplicate URLs:", df["URL"].duplicated().sum())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df[TARGET_COLUMN].value_counts())


def clean_data(df):
    """
    Clean the dataset.

    Steps:
    - Remove duplicate URLs
    - Remove rows containing missing values
    """
    df = df.copy()

    original_rows = df.shape[0]

    # Remove duplicate URLs
    df = df.drop_duplicates(subset=["URL"])

    # Remove missing-value rows, if any
    df = df.dropna()

    cleaned_rows = df.shape[0]

    print("\n===== AFTER CLEANING =====")
    print("Original rows:", original_rows)
    print("Rows after cleaning:", cleaned_rows)
    print("Rows removed:", original_rows - cleaned_rows)

    return df


def prepare_features(
    df,
    remove_similarity_index=False
):
    """
    Separate the features X from the target y.

    Text columns are removed because the Random Forest
    model currently uses numeric/binary features only.

    If remove_similarity_index=True,
    URLSimilarityIndex is removed for validation testing.
    """

    columns_to_drop = TEXT_COLUMNS + [TARGET_COLUMN]

    if remove_similarity_index:
        columns_to_drop.append("URLSimilarityIndex")

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    print("\n===== FEATURE PREPARATION =====")
    print("Number of input features:", X.shape[1])

    if remove_similarity_index:
        print("URLSimilarityIndex removed: YES")
    else:
        print("URLSimilarityIndex removed: NO")

    return X, y


def split_data(X, y):
    """
    Split the dataset into:
    - 80% training data
    - 20% testing data

    Stratification preserves the class distribution.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(
    remove_similarity_index=False
):
    """
    Complete preprocessing pipeline.

    Returns:
        X_train
        X_test
        y_train
        y_test
    """

    # 1. Load dataset
    df = load_data()

    # 2. Inspect original dataset
    inspect_data(df)

    # 3. Clean dataset
    df = clean_data(df)

    # 4. Prepare input features and target
    X, y = prepare_features(
        df,
        remove_similarity_index=remove_similarity_index,
    )

    # 5. Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    print("\n===== PREPROCESSED DATA =====")
    print("Number of features:", X.shape[1])
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    print("\n===== TRAIN TARGET DISTRIBUTION =====")
    print(y_train.value_counts())

    print("\n===== TEST TARGET DISTRIBUTION =====")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()