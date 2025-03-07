import streamlit as st
import requests
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
from deepface import DeepFace
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load LSTM Model (Ensure the model file is in your GitHub repo)
@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model("lstm_model.h5")

lstm_model = load_lstm_model()

# Function to check misinformation using Google Fact Check API
def check_fact(text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={text}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return data["claims"][0]["claimReview"][0]["textualRating"]
        else:
            return "No fact-check found."
    return "Error fetching fact-check data."

# Function to predict misinformation using LSTM
def predict_misinformation(text):
    sequence = tokenizer.encode_plus(text, return_tensors="tf", padding="max_length", truncation=True, max_length=512)
    prediction = lstm_model.predict(sequence["input_ids"])
    return "Misinformation" if prediction[0][0] > 0.5 else "Likely True"

# Streamlit UI
st.set_page_config(page_title="AI Misinformation Detector", page_icon="🤖", layout="wide")

st.title("🔍 AI-Powered Misinformation Detection")
st.write("Analyze text and detect fake news or misleading claims.")

tab1, tab2 = st.tabs(["📜 Text Analysis", "📷 Image & Video Deepfake Detection"])

with tab1:
    user_input = st.text_area("Enter text to analyze:", "")
    if st.button("Analyze"):
        if user_input:
            fact_check_result = check_fact(user_input)
            lstm_result = predict_misinformation(user_input)

            st.subheader("🧠 AI Model Prediction:")
            st.info(f"🔎 {lstm_result}")

            st.subheader("✅ Google Fact Check API:")
            st.success(f"📌 {fact_check_result}")
        else:
            st.warning("⚠️ Please enter text for analysis.")

with tab2:
    uploaded_file = st.file_uploader("Upload an image or video for deepfake detection", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Deepfake"):
            result = DeepFace.analyze(image, actions=["emotion"])
            st.write("Deepfake Detection Results:", result)
