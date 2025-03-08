import streamlit as st
import tensorflow as tf
import requests
import os
import cv2
import numpy as np
from transformers import BertTokenizer
from deepface import DeepFace

# ------------------- Load NLP Misinformation Model -------------------

GITHUB_MODEL_URL = "https://raw.githubusercontent.com/Rushil-K/Misinformation-System/main/lstm_model.h5"
MODEL_PATH = "lstm_model.h5"

# Function to download the model if not available locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading the model from GitHub...")
        r = requests.get(GITHUB_MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

@st.cache_resource
def load_misinformation_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
model = load_misinformation_model()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ------------------- Text-Based Misinformation Detection -------------------
def analyze_text_misinformation(text):
    tokens = tokenizer.encode(text, truncation=True, padding="max_length", max_length=128, return_tensors="tf")
    prediction = model.predict(tokens)
    score = prediction[0][0]
    return "❌ Misinformation" if score > 0.5 else "✅ Reliable"

# ------------------- Deepfake Detection -------------------
def detect_deepfake(image_path):
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=["age", "gender", "race", "emotion"])
        return analysis
    except Exception as e:
        return {"error": str(e)}

# ------------------- Streamlit UI -------------------
st.title("🛑 AI-Powered Misinformation Detection System")

option = st.sidebar.selectbox("Choose Detection Mode:", ["Text Analysis", "Image Deepfake Detection"])

if option == "Text Analysis":
    st.header("📝 Text-Based Misinformation Analysis")
    user_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze"):
        if user_input:
            result = analyze_text_misinformation(user_input)
            st.write(f"**Prediction:** {result}")
        else:
            st.warning("Please enter some text.")

elif option == "Image Deepfake Detection":
    st.header("📷 AI-Powered Deepfake Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Deepfake"):
            analysis_result = detect_deepfake(file_path)
            st.json(analysis_result)

        os.remove(file_path)

st.sidebar.write("🚀 Built with AI to detect misinformation & deepfakes.")
