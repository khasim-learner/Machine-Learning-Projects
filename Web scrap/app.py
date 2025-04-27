import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define headers to look like a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0.0.0 Safari/537.36"
}

def scrape_reviews(base_url, pages=5):
    reviews = []
    ratings = []
    
    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            st.warning(f"Failed to retrieve page {page}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        review_blocks = soup.find_all('article')

        for review in review_blocks:
            text_tag = review.find('p')
            review_text = text_tag.text.strip() if text_tag else None
            
            rating_tag = review.find('div', {"data-service-review-rating": True})
            if rating_tag:
                rating = int(rating_tag['data-service-review-rating'])
            else:
                rating = None

            if review_text and rating:
                reviews.append(review_text)
                ratings.append(rating)

        time.sleep(random.uniform(1, 2))  # Polite delay

    data = pd.DataFrame({'review': reviews, 'rating': ratings})
    return data

def train_model(data):
    data['label'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)
    data = data.dropna()

    X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, vectorizer, report

def predict_sentiment(model, vectorizer, text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)
    return "Positive" if pred[0] == 1 else "Negative"

# Streamlit UI
st.title("🌟 Web Scraper + Sentiment Classifier App")

st.markdown("""
This app scrapes reviews from **Trustpilot** and predicts if they are **Positive** or **Negative** based on a Machine Learning model!
""")

base_url = st.text_input("Enter Trustpilot review page URL:", "https://www.trustpilot.com/review/www.apple.com")
pages_to_scrape = st.slider("Number of pages to scrape:", 1, 10, 3)

if st.button("Scrape and Train Model"):
    with st.spinner("Scraping reviews and training model..."):
        data = scrape_reviews(base_url, pages=pages_to_scrape)
        if len(data) == 0:
            st.error("No reviews found. Try again!")
        else:
            st.success(f"Scraped {len(data)} reviews!")

            st.dataframe(data.head())

            model, vectorizer, report = train_model(data)

            st.subheader("Model Performance Metrics:")
            st.json(report)

            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.success("Model trained and ready!")

# Allow user to input a review and predict
if "model" in st.session_state and "vectorizer" in st.session_state:
    st.subheader("Test the model:")
    user_review = st.text_area("Enter a review text to predict sentiment:")
    
    if st.button("Predict Sentiment"):
        result = predict_sentiment(st.session_state.model, st.session_state.vectorizer, user_review)
        st.info(f"Predicted Sentiment: **{result}**")
