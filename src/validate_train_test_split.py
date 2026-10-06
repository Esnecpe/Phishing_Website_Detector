from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

# IMPORTANT: same cleaning step as data_preprocessing.py
df = df.drop_duplicates(subset=["URL"])

print("===== TRAIN / TEST OVERLAP CHECK =====")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

train_urls = set(train_df["URL"])
test_urls = set(test_df["URL"])

url_overlap = train_urls.intersection(test_urls)

print("\n===== URL OVERLAP =====")
print("Unique URLs in training:", len(train_urls))
print("Unique URLs in testing:", len(test_urls))
print("URLs appearing in BOTH:", len(url_overlap))

if len(url_overlap) == 0:
    print("GOOD: No exact URL overlap found.")
else:
    print("WARNING: Some URLs appear in both training and testing.")

train_domains = set(train_df["Domain"])
test_domains = set(test_df["Domain"])

domain_overlap = train_domains.intersection(test_domains)

print("\n===== DOMAIN OVERLAP =====")
print("Unique domains in training:", len(train_domains))
print("Unique domains in testing:", len(test_domains))
print("Domains appearing in BOTH:", len(domain_overlap))

percentage = (
    len(domain_overlap) / len(test_domains) * 100
    if len(test_domains) > 0
    else 0
)

print(
    f"Percentage of testing domains also seen in training: "
    f"{percentage:.2f}%"
)

print("\n===== DATASET DUPLICATION =====")
print("Duplicate URLs:", df["URL"].duplicated().sum())
print("Repeated domains:", df["Domain"].duplicated().sum())