import streamlit as st
import tensorflow as tf
import numpy as np
import os
from transformers import AutoTokenizer

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# Define model path
MODEL_PATH = "lstm_model.h5"

# Load NLP Model
@st.cache_resource
def load_text_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Error: Text model file not found.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

# Process Text Input
def analyze_text(model, tokenizer, text):
    if model is None:
        return "Error: Model not loaded."

    # Tokenize input text
    tokens = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)

    # Convert tokenized input into embeddings (to match LSTM input shape)
    embedding_layer = tf.keras.layers.Embedding(input_dim=30522, output_dim=128)
    embedded_input = embedding_layer(tokens["input_ids"])

    # Ensure input is 3D: (batch_size, timesteps, features)
    embedded_input = tf.reshape(embedded_input, (embedded_input.shape[0], embedded_input.shape[1], 128))

    # Get prediction
    prediction = model.predict(embedded_input)
    labels = ['False', 'Half-True', 'Mostly-True', 'True', 'Barely-True', 'Pants-on-Fire']
    
    return labels[np.argmax(prediction)]

# Streamlit UI
st.title("Misinformation Text Analysis")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze Text"):
    text_model = load_text_model()
    tokenizer = load_tokenizer()
    result = analyze_text(text_model, tokenizer, user_input)
    st.write(f"Prediction: {result}")
