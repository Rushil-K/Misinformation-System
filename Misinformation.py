import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import onnxruntime

# Try to import OpenCV, handle error if missing
try:
    import cv2
except ImportError:
    st.error("OpenCV (cv2) is not installed. Please install it using `pip install opencv-python`.")

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# Define paths
MODEL_PATH = "lstm_model.h5"
DEEPFAKE_MODEL_PATH = "models/deepfake_detector.onnx"

# Load NLP Model
def load_text_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Text model file not found.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# Load Tokenizer
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Deepfake Detection Model
def load_deepfake_model():
    if not os.path.exists(DEEPFAKE_MODEL_PATH):
        st.error("Deepfake model file not found.")
        return None
    return onnxruntime.InferenceSession(DEEPFAKE_MODEL_PATH)

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

# Process Image for Deepfake Detection
def analyze_image(model, image):
    if model is None:
        return "Error: Model not loaded."
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0).numpy()
    result = model.run(None, {model.get_inputs()[0].name: image})
    return "Deepfake Detected" if result[0][0] > 0.5 else "Real Image"

# Streamlit UI
st.title("Misinformation Detection System")

st.sidebar.header("Choose an option:")
option = st.sidebar.selectbox("Select Input Type", ["Text Analysis", "Image Deepfake Detection"])

if option == "Text Analysis":
    text_model = load_text_model()
    tokenizer = load_tokenizer()
    user_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text"):
        result = analyze_text(text_model, tokenizer, user_input)
        st.write(f"Prediction: {result}")

elif option == "Image Deepfake Detection":
    deepfake_model = load_deepfake_model()
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            result = analyze_image(deepfake_model, image)
            st.write(f"Prediction: {result}")
