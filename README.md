# Phishing Website Detector

A machine learning class project that detects whether a URL is likely to be **phishing** or **legitimate**.

The project uses the **PhiUSIIL Phishing URL Dataset** and compares machine learning models such as **Random Forest** and **Logistic Regression**.

The final application uses a **URL-only Random Forest deployment model** so that predictions can be made directly from the URL entered by the user.

---

## Repository Structure

```text
Phishing_Website_Detector/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── url_random_forest.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── gui.py
│   ├── model_test_LogisticRegression.py
│   ├── model_test_randomforest.py
│   ├── phishing_predictor.py
│   └── train_url_deployment_model.py
│
├── .gitignore
├── README.md
└── requirements.txt