# app.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index and reverse mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained RNN model
model = load_model('simple_rnn_imdb.h5')
max_len = 500

# Function to decode review (for future use)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input and identify unknown words
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    unknown_words = []

    for word in words:
        if word in word_index and word_index[word] < 10000:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)  # OOV token
            unknown_words.append(word)

    if unknown_words:
        st.info(f"The following word(s) are new to the model and were not recognized: {', '.join(set(unknown_words))}")

    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# ----------------- Streamlit UI -----------------

# Title and Instructions
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title('IMDB Movie Review Sentiment Classifier')
st.markdown("""
Enter a short movie review in the box below. The model will analyze the **sentiment** of your review and classify it as:
- Positive
- Negative

This tool uses a simple RNN model trained on a portion of the IMDB movie reviews dataset.
""")

# Sidebar - Info & Help
with st.sidebar:
    st.header("How it Works")
    st.markdown("""
    - The model is trained using a subset of the IMDB dataset with only the **10,000 most frequent words**.
    - It uses a **SimpleRNN** layer with `tanh` activation.
    - Words not in the model’s vocabulary are treated as unknown.
    """)
    st.warning("""
    This model may not perform accurately on complex or unusual reviews due to limited vocabulary and training data.
    """)

# Text Input
user_input = st.text_area('Enter Your Movie Review:', height=150, placeholder="e.g. The movie was terrible but the songs were good...")

if st.button('Analyze Sentiment'):

    if not user_input.strip():
        st.warning("Please enter a non-empty review.")
    else:
        # Preprocess and Predict
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        prediction_score = float(prediction[0][0])  # ensure proper float

        if prediction_score > 0.5:
            sentiment = 'Positive'
            confidence = prediction_score
        else:
            sentiment = 'Negative'
            confidence = 1 - prediction_score

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {confidence:.2f}")

        if 0.4 < prediction_score < 0.6:
            st.info("This review appears to have mixed or ambiguous sentiment.")
else:
    st.caption("Submit a review above to get its sentiment prediction.")
