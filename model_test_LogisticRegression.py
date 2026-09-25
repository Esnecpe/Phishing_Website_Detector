import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Scaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

#need to test more i feel unconfident that my code is right lol

# 1. Load dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Remove non-numeric/text columns for the first version
drop_columns = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title"
]

X = df.drop(columns=drop_columns + ["label"])
y = df["label"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#4. Need to scale inputs: Lecture from 9/22


# 5. Create and train Logistic Regression
model = LogisticRegression(
    #Dont know what to put for now, will find out later
    solver="lbfgs",
    C=1.0,
    max_iter=2000
)

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Phishing Precision:", precision_score(y_test, y_pred, pos_label=0))
print("Phishing Recall:", recall_score(y_test, y_pred, pos_label=0))
print("Phishing F1 Score:", f1_score(y_test, y_pred, pos_label=0))

print("\nConfusion Matrix (rows=actual, columns=predicted; order: phishing, legitimate):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    digits=4
))
