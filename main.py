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

# Decode function (not actively used)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess function
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

    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review, unknown_words

# Streamlit setup
st.set_page_config(page_title="🎬 IMDB Sentiment Classifier", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Classifier")

st.markdown("""
Welcome! This app analyzes the **sentiment** of a movie review and classifies it as:
- ✅ **Positive**
- ❌ **Negative**

It uses a **Simple RNN** trained on a portion of the IMDB dataset, and understands only the 10,000 most common words.
""")

# Input
user_input = st.text_area("✏️ Enter Your Movie Review:", height=150, placeholder="e.g. The movie was terrible but the songs were good...")

# On click
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a non-empty review.")
    else:
        try:
            preprocessed_input, unknown_words = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            prediction_score = float(prediction[0][0])

            if prediction_score > 0.5:
                sentiment = "Positive 😄"
                confidence = prediction_score
            else:
                sentiment = "Negative 😞"
                confidence = 1 - prediction_score

            st.success(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence Score:** `{confidence:.2f}`")

            if unknown_words:
                st.info("📌 New words not recognized by the model: " + ", ".join(set(unknown_words)))

            if 0.4 < prediction_score < 0.6:
                st.info("🤔 This review appears to have **mixed or ambiguous sentiment**.")

        except Exception as e:
            st.error("❌ An unexpected error occurred during prediction. Please try a simpler review.")
else:
    st.caption("⬆️ Submit a review above to see its sentiment prediction.")
