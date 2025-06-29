import streamlit as st
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)

st.title("✈️ Airline Tweet Sentiment Classifier")
st.write("Enter a tweet below to predict whether it's Positive, Neutral, or Negative.")

tweet = st.text_area("Enter a tweet here:")

if st.button("Predict Sentiment"):
    cleaned = clean_text(tweet)
    vector = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vector)[0]
    sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[pred]
    st.success(f"Predicted Sentiment: {sentiment}")
