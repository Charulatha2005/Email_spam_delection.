import pandas as pd
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df.iloc[:, :2]
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    return text

df["message"] = df["message"].apply(clean_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)


def predict_email(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    return "SPAM" if result == 1 else "HAM"

# Example
email = "Win free cash prize now"
print("\nEmail Prediction:", predict_email(email))
