# Phishing Website Detector

CPSC 483 - Introduction to Machine Learning, Fall 2026, California State University, Fullerton
**Team:** Adrian Hernandez (Random Forest) and Jesriel Shen (Logistic Regression)

A machine learning project that classifies a URL as **phishing** or **legitimate**.
Two models, Random Forest and Logistic Regression, are trained and compared on the same dataset.
A Tkinter GUI checks any URL with both models side by side.

---

## Dataset

**PhiUSIIL Phishing URL Dataset** (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/967

- 235,795 URLs (134,850 legitimate, 100,945 phishing), 56 columns
- Labels: `0 = Phishing`, `1 = Legitimate`
- Citation: Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.1016/j.cose.2023.103545
- License: CC BY 4.0

The CSV is included in `data/`.

---

## Repository Structure

```text
Phishing_Website_Detector/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
├── figures/                               # plots created by the model scripts
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importance.png
│   ├── rf_roc_curve.png
│   ├── lr_coefficients.png
│   ├── lr_precision_recall_curve.png
│   └── lr_roc_curve.png
├── models/
│   ├── random_forest_model.pkl            # 49-feature Random Forest
│   ├── url_random_forest.pkl              # 15-feature URL-only Random Forest (GUI)
│   └── url_logistic_model.pkl             # 15-feature URL-only Logistic Regression (GUI)
├── src/
│   ├── data_preprocessing.py              # shared cleaning + train/test split
│   ├── model_test_randomforest.py         # Random Forest evaluation (Adrian)
│   ├── model_test_LogisticRegression.py   # Logistic Regression evaluation (Jesriel)
│   ├── train_final_random_forest.py       # trains and saves random_forest_model.pkl
│   ├── train_url_deployment_model.py      # trains and saves both URL-only models
│   ├── phishing_predictor.py              # URL feature extraction + predictors
│   └── gui.py                             # Tkinter app
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.11+ (Tkinter comes with Python on Windows and macOS).

```bash
git clone https://github.com/Esnecpe/Phishing_Website_Detector.git
cd Phishing_Website_Detector
pip install -r requirements.txt
```

Run every command below from the project root folder.

---

## How to Run

| Step | Command | What it does |
|---|---|---|
| 1 | `python src/data_preprocessing.py` | Prints dataset info, cleaning results, and input checks |
| 2 | `python src/model_test_randomforest.py` | Trains and evaluates Random Forest, saves 3 `rf_*` figures (about 1 minute) |
| 3 | `python src/model_test_LogisticRegression.py` | Trains and evaluates Logistic Regression, saves 3 `lr_*` figures |
| 4 | `python src/train_final_random_forest.py` | Trains the 49-feature Random Forest and saves `models/random_forest_model.pkl` |
| 5 | `python src/train_url_deployment_model.py` | Trains both URL-only models and saves them to `models/` |
| 6 | `python src/phishing_predictor.py` | Tests one URL in the terminal |
| 7 | `python src/gui.py` | Opens the GUI app |

Steps 4 and 5 are optional because the trained models are already included in `models/`.

---

## Data Preprocessing

Done in `src/data_preprocessing.py` and shared by both models:

1. Removed **425 duplicate URLs** (235,795 → 235,370 rows). No missing values were found.
2. Dropped text columns: `FILENAME`, `URL`, `Domain`, `TLD`, `Title`.
3. Dropped `URLSimilarityIndex`, because it caused 100% accuracy and may leak the label.
4. Stratified 80/20 split with `random_state=42` → **188,296 training / 47,074 testing**.
5. Checked for invalid numbers (none found) and reported outliers (kept, since unusual values can be real phishing signals).

Result: **49 input features**.

---

## Results (49 features, same data split)

### Random Forest (Adrian Hernandez)

`RandomForestClassifier(n_estimators=100, random_state=42)`

| Metric | Phishing (0) | Legitimate (1) |
|---|---|---|
| Precision | 1.000000 | 0.999889 |
| Recall | 0.999851 | 1.000000 |
| F1-score | 0.999925 | 0.999944 |

- Accuracy: **0.999936** (3 mistakes out of 47,074)
- ROC-AUC: **1.0000**
- 5-fold cross-validation accuracy: **0.999857 ± 0.000046**
- Hyperparameter tuning (RandomizedSearchCV) did not improve the test score, so the default settings were kept.

Figures: `rf_confusion_matrix.png`, `rf_feature_importance.png`, `rf_roc_curve.png`

### Logistic Regression (Jesriel Shen)

`Pipeline(StandardScaler, LogisticRegression(C=1.0, max_iter=2000))`. The scaler is inside the pipeline, so it is fit on training data only.

| Metric | Phishing (0) | Legitimate (1) |
|---|---|---|
| Precision | 0.999353 | 0.999148 |
| Recall | 0.998856 | 0.999518 |
| F1-score | 0.999104 | 0.999333 |

- Accuracy: **0.999235**
- ROC-AUC: **0.999996**
- 5-fold cross-validation accuracy: **0.999230 ± 0.000130**

Figures: `lr_coefficients.png`, `lr_precision_recall_curve.png`, `lr_roc_curve.png`

### Comparison

| Model | Accuracy | Phishing F1 | ROC-AUC | 5-fold CV accuracy |
|---|---|---|---|---|
| Random Forest | **0.999936** | **0.999925** | **1.0000** | 0.999857 ± 0.000046 |
| Logistic Regression | 0.999235 | 0.999104 | 0.999996 | 0.999230 ± 0.000130 |

Both models score above 99.9%. Random Forest is slightly better because it can learn non-linear combinations of features, while Logistic Regression draws a single linear boundary.
The most important inputs for both models describe the webpage content (for example `LineOfCode`, `NoOfSelfRef`, `NoOfJS`), which separate the two classes almost perfectly in this dataset.

---

## The GUI App

`src/gui.py` checks a URL with **both URL-only models** at the same time.

1. Enter a URL and click **Check both** (or press Enter).
2. Each model shows its own result, phishing risk bar, and risk percentage:

| Phishing risk | Result | Color |
|---|---|---|
| 0% to 30% | LEGITIMATE | Green |
| 30% to 70% | SUSPICIOUS | Yellow |
| 70% to 100% | PHISHING | Red |

3. The app also shows URL details, simple warning signs, and a history of checked URLs.

The SUSPICIOUS level is only a display range in the GUI. The models themselves still predict phishing or legitimate.
The app only reads the URL text. It never opens the website, so suspicious links can be checked safely.

### URL-only models (15 URL features)

The 49-feature models need webpage content features that were computed by the dataset authors and cannot be reproduced from a live URL.
So the GUI uses two separate models trained on **15 features calculated by our own code from the URL text**:

`URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`

The same function (`extract_url_features` in `phishing_predictor.py`) is used for training and for predictions in the GUI.
Before the features are calculated, a URL without `http://` or `https://` gets `https://` added, and a single `/` at the end of the domain is removed (so `example.com/` and `example.com` are treated the same).

| URL-only model | Accuracy | Phishing precision | Phishing recall | Phishing F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.967902 | 0.983412 | 0.940708 | 0.961586 | 0.987465 |
| Logistic Regression | 0.952692 | 0.980590 | 0.907183 | 0.942459 | 0.980637 |

### Why the trailing "/" is removed

In the dataset, 33% of phishing URLs end with a `/` after the domain, but no legitimate URL does.
An earlier version kept that `/` and scored 99.5%, partly because the model used it as a shortcut.
Removing it makes the comparison fairer, and accuracy dropped to 96.8%.

### Known limitations

Every legitimate URL in the dataset is a homepage written as `https://www.domain` with nothing after it.
The URL-only models learned part of this pattern, especially Random Forest:

| URL | Random Forest | Logistic Regression |
|---|---|---|
| `https://www.google.com` | LEGITIMATE (1.18%) | LEGITIMATE (2.21%) |
| `google.com` (no `www.`) | PHISHING (100%) | LEGITIMATE (4.25%) |
| `https://www.google.com/maps` (has a path) | PHISHING (98.13%) | SUSPICIOUS (60.41%) |

- **No `www.`** or **a path after the domain** can raise the risk score for real websites.
- **HTTP:** URLs starting with `http://` are usually flagged, because every legitimate URL in the dataset uses HTTPS (`IsHTTPS` is the most important feature, 0.431).
- A possible improvement is to use features from the domain name only, or to train with legitimate URLs that include paths from another source.

This tool is an educational prototype and should not replace professional security tools.

---

## Libraries

Python, pandas, NumPy, scikit-learn, matplotlib, joblib, Tkinter (see `requirements.txt` for versions).

## Acknowledgments

- Dataset by Arvind Prasad and Shalini Chandra (see citation above).
- ChatGPT (OpenAI) and Claude (Anthropic) were used for help with code, debugging, and documentation.