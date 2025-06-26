# IMDB Movie Review Sentiment Classifier

This is a Streamlit web application that performs sentiment analysis on movie reviews using a Recurrent Neural Network (RNN) trained on the IMDB dataset.

## Project Overview

The application allows users to input a movie review in plain English and receive a sentiment prediction: **Positive** or **Negative**, along with a confidence score.

The model is trained using Keras on the IMDB dataset and uses a `SimpleRNN` layer with `tanh` activation. The app provides an easy-to-use interface for real-time sentiment classification.

---

## Features

- Classifies text reviews into Positive or Negative sentiment.
- Displays confidence score of prediction.
- Handles unknown or rare words with an out-of-vocabulary token.
- Provides a clear explanation for mixed or ambiguous predictions.
- Simple and responsive UI built with Streamlit.

---

## Model Architecture

- Embedding Layer: Converts word indices into dense vectors.
- SimpleRNN Layer: Captures sequential information from reviews.
- Dense Layer with Sigmoid: Outputs probability for binary classification.

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- Streamlit
- IMDB Dataset (from `tensorflow.keras.datasets`)
- NumPy

---

## 🔧 Setup Instructions

1. Clone the repository or copy the project files.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
