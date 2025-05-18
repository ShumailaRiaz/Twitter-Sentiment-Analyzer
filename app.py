import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import base64
import os
import gdown # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Load the CSV dataset from Google Drive
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1xV0g5F1tCX85xrBeVjxhXbNrg3n2-bM7"
    output = "tweetdataset.csv"

    # Download only if file doesn't exist
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    try:
        df = pd.read_csv(output, encoding='latin-1')
        df['username'] = df['username'].astype(str).str.lower()
        return df
    except Exception as e:
        st.error(f"Failed to read the dataset: {e}")
        return pd.DataFrame(columns=['username', 'tweet'])

# Set background image
def set_background(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found. Proceeding without background.")

# Preprocess and predict sentiment
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)
    return "Negative" if prediction[0] == 0 else "Positive"

# Create styled card for each tweet
def create_card(tweet_text, sentiment):
    color = "#22c55e" if sentiment == "Positive" else "#ef4444"
    emoji = "😊" if sentiment == "Positive" else "😠"
    card_html = f"""
    <div style="background-color:{color}; padding: 20px; border-radius: 12px; margin: 15px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <h4 style="color:white;">{emoji} {sentiment} Sentiment</h4>
        <p style="color:white; font-size: 16px;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main application
def main():
    st.set_page_config(page_title="Legendary Twitter Sentiment Analyzer", layout="wide")

    # Set background
    set_background("bg18.jpeg")

    # Title and subtitle
    st.markdown('<h1 style="font-size: 40px; color: #FFD700; font-weight: 800; margin-bottom: 10px;">⚡Twitter Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 18px; color: #ffffff; margin-bottom: 30px;">Analyze how people feel—one tweet at a time.</p>', unsafe_allow_html=True)

    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    data = load_data()

    # Sidebar options
    st.sidebar.title("Choose Input Method")
    option = st.sidebar.radio("", ["📜 Input text", "🐦 Fetch tweets from Dataset"])

    if option == "📜 Input text":
        text_input = st.text_area("✏️ Enter your text below")
        if st.button("🔍 Analyze"):
            if text_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                card_html = create_card(text_input, sentiment)
                st.markdown(card_html, unsafe_allow_html=True)

    elif option == "🐦 Fetch tweets from Dataset":
        username = st.text_input("Enter Twitter Username (without @)").lower().strip()
        if st.button("📥 Fetch Tweets"):
            if username == "":
                st.warning("Please enter a username.")
            else:
                user_tweets = data[data['username'] == username]
                if user_tweets.empty:
                    st.warning(f"No tweets found for username: {username}")
                else:
                    for tweet in user_tweets['tweet']:
                        sentiment = predict_sentiment(tweet, model, vectorizer, stop_words)
                        card_html = create_card(tweet, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
