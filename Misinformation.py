import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

# Define paths
MODEL_PATH = "models/lstm_model.h5"

# Load NLP Model
@st.cache_resource
def load_text_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

# Process Text Input
def analyze_text(model, tokenizer, text):
    if model is None:
        return "Error: Model not loaded."
    
    tokens = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    input_ids = tokens['input_ids']
    
    # Ensure correct shape: (batch_size, timesteps, features)
    input_ids = tf.cast(input_ids, tf.float32)  # Convert to float for LSTM compatibility
    input_ids = tf.expand_dims(input_ids, -1)  # Ensure correct input shape for LSTM
    
    prediction = model.predict(input_ids)
    
    labels = ['False', 'Half-True', 'Mostly-True', 'True', 'Barely-True', 'Pants-on-Fire']
    return labels[np.argmax(prediction)]

# Streamlit UI
st.title("Misinformation Detection System (Text Analysis)")

# Load models
text_model = load_text_model()
tokenizer = load_tokenizer()

# User input
user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze Text"):
    result = analyze_text(text_model, tokenizer, user_input)
    st.write(f"Prediction: {result}")
