import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from collections import Counter

with open('model.joblib', 'rb') as file:
    model = joblib.load(file)

with open('reference_words.joblib', 'rb') as file:
    reference_words = joblib.load(file)

def preprocess_text(news, lower_case=True):
    if lower_case:
        news = news.lower()
    news = re.sub(r'\d+', 'Number', news)
    news = re.sub(r'[^\w\s]', '', news)
    return news

def transform_input(text, reference_words):
    word_counts = Counter(text.split())
    data = {word: 0 for word in reference_words}
    for word, count in word_counts.items():
        if word in data:
            data[word] = count
    df = pd.DataFrame([data], columns=reference_words)
    return df

st.title('Sentiment Analysis Tool')
st.write('Enter a news headline to predict its sentiment:')

# User input
user_input = st.text_input("News Headline", "")

if st.button('Predict Sentiment'):
    if not user_input:
        st.write("Please, enter something.")
    else:
        preprocessed_text = preprocess_text(user_input)
        transformed_df = transform_input(preprocessed_text, reference_words)
        prediction = model.predict(transformed_df)
        st.write(f'The predicted sentiment is: {prediction[0]}')