# Phishing Website Detector

Machine-learning project using the PhiUSIIL Phishing URL Dataset and a Random Forest classifier.

## Repository structure

```text
Phishing_Website_Detector/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
├── models/
│   ├── deployment_random_forest.pkl
│   └── random_forest_model.pkl
├── src/
│   ├── __init__.py
│   ├── cross_validation_randomforest.py
│   ├── data_preprocessing.py
│   ├── feature_extractor.py
│   ├── gui.py
│   ├── model_test_randomforest.py
│   ├── phishing_predictor.py
│   ├── random_forest_tuning.py
│   ├── train_deployment_random_forest.py
│   ├── train_final_random_forest.py
│   └── validate_train_test_split.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Main workflows

Train the 49-feature academic Random Forest model:

```bash
python src/train_final_random_forest.py
```

Train the 41-feature deployment Random Forest model:

```bash
python src/train_deployment_random_forest.py
```

Test the live feature extractor:

```bash
python src/feature_extractor.py
```

Run a terminal prediction with the deployment model:

```bash
python src/phishing_predictor.py
```

Run the GUI:

```bash
python src/gui.py
```

## Model files

- `models/random_forest_model.pkl` — 49-feature model used for the main academic Random Forest analysis.
- `models/deployment_random_forest.pkl` — 41-feature model intended for the live application because those features can be reproduced by `feature_extractor.py`.

The current label mapping is:

- `0` = phishing
- `1` = legitimate
