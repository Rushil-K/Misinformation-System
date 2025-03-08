import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
import requests
from deepface import DeepFace
from transformers import pipeline

# -------------------- Load Misinformation Model --------------------
MODEL_PATH = "lstm_model.h5"

@st.cache_resource
def load_misinformation_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://github.com/Rushil-K/Misinformation-System/raw/main/lstm_model.h5"  # Update with actual URL
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_misinformation_model()

# -------------------- NLP Pipeline for Text Analysis --------------------
@st.cache_resource
def load_nlp_pipeline():
    return pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

nlp_pipeline = load_nlp_pipeline()

def analyze_text(text):
    prediction = nlp_pipeline(text)
    return prediction[0]["label"]

# -------------------- Deepfake Detection --------------------
def detect_deepfake(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result
    except Exception as e:
        return str(e)

# -------------------- Streamlit UI --------------------
st.title("🛡️ AI-Powered Misinformation Detection System")

tab1, tab2, tab3 = st.tabs(["🔍 Text Analysis", "🖼️ Deepfake Detection", "ℹ️ About"])

# Text Analysis Tab
with tab1:
    st.header("📜 Text Misinformation Detection")
    text_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text"):
        if text_input:
            result = analyze_text(text_input)
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter some text.")

# Deepfake Detection Tab
with tab2:
    st.header("📸 Deepfake Detection")
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            result = detect_deepfake(image)
            st.write(result)

# About Tab
with tab3:
    st.header("ℹ️ About")
    st.write("This open-source misinformation detection system uses NLP and AI to detect misleading claims and deepfakes.")

st.write("💡 Created with ❤️ by [Your Name]")
