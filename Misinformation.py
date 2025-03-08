import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import os

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# Define model path
MODEL_PATH = "lstm_model.h5"

# Load NLP Model
def load_text_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Text model file not found.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# Load Tokenizer
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

# Process Text Input
def analyze_text(model, tokenizer, text):
    if model is None:
        return "Error: Model not loaded."
    
    tokens = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    input_ids = tokens['input_ids']
    input_ids = tf.reshape(input_ids, (input_ids.shape[0], input_ids.shape[1], 1))  # Ensure correct shape for LSTM
    prediction = model.predict(input_ids)
    
    labels = ['False', 'Half-True', 'Mostly-True', 'True', 'Barely-True', 'Pants-on-Fire']
    return labels[np.argmax(prediction)]

# Streamlit UI
st.title("Misinformation Text Analysis")

text_model = load_text_model()
tokenizer = load_tokenizer()

user_input = st.text_area("Enter text to analyze:")
if st.button("Analyze Text"):
    result = analyze_text(text_model, tokenizer, user_input)
    st.write(f"Prediction: {result}")
