import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

def train_and_save_model():
    # Load your dataset
    df = pd.read_csv("C:\\Users\\home\\Downloads\\combined_emails_with_natural_pii.csv")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['email'])
    y = df['type']
    model = MultinomialNB()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

def classify_email(email):
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = vectorizer.transform([email])
    return model.predict(X)[0]
