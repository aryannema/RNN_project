# Step 1: Import Libraries and Load the Model
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

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# ----------------- Streamlit UI -----------------

# Title and Instructions
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title('🎬 IMDB Movie Review Sentiment Classifier')
st.markdown("""
Enter a short movie review in the box below. The model will analyze the **sentiment** of your review and classify it as:
- ✅ Positive
- ❌ Negative

It uses a simple RNN trained on the **IMDB dataset**.
""")

# Sidebar - Info & Help
with st.sidebar:
    st.header("ℹ️ How it Works")
    st.markdown("""
    - The model processes up to **500 words** per review.
    - It uses a **SimpleRNN** with `tanh` activation.
    - Only common words (top 10,000) from the IMDB dataset are recognized.
    - Unknown or rare words will be replaced with a special token.
    """)
    st.info("Model trained using Keras + IMDB dataset.")

# Text Input
user_input = st.text_area('✏️ Enter Your Movie Review:', height=150, placeholder="e.g. The movie was terrible but the songs were good...")

if st.button('🔍 Analyze Sentiment'):

    if not user_input.strip():
        st.warning("Please enter a non-empty review.")
    else:
        # Preprocess and Predict
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        prediction_score = float(prediction[0][0])  # ensure proper float casting

        # Classify based on threshold
        if prediction_score > 0.5:
            sentiment = 'Positive 😄'
            confidence = prediction_score
        else:
            sentiment = 'Negative 😞'
            confidence = 1 - prediction_score

        # Display result
        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** `{confidence:.2f}`")

        # Optional explanation
        if 0.4 < prediction_score < 0.6:
            st.info("This review has **mixed or ambiguous sentiment** — the model was slightly uncertain.")
else:
    st.caption("⬆️ Submit a review above to see its sentiment prediction.")
