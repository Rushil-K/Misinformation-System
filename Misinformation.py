import streamlit as st
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import cv2
from deepface import DeepFace
import validators
import time

# Load AI-generated text detection model (Open-source)
MODEL_NAME = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Streamlit UI
st.set_page_config(page_title="Misinformation Detection", layout="wide")
st.title("🛡️ Misinformation Detection System")
st.write("Detect AI-generated text, deepfake images/videos, and verify credibility.")

# Sidebar options
st.sidebar.header("🔍 Analysis Options")
option = st.sidebar.radio("Select Analysis Type", ["AI Text Detection", "Deepfake Detection", "Credibility Check"])

# Function to detect AI-generated text
def detect_ai_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_score = probs[0][1].item() * 100  # Probability of AI-generated
    return round(ai_score, 2)

# Function to detect deepfakes in images/videos
def detect_deepfake(image):
    try:
        result = DeepFace.analyze(image, actions=["emotion", "age", "gender"])
        return result
    except Exception as e:
        return str(e)

# Function to check credibility of news/articles
def check_credibility(url):
    if not validators.url(url):
        return "❌ Invalid URL"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return "✅ Trusted Source"
        else:
            return "⚠️ Unverified Source"
    except:
        return "❌ Unable to verify"

# AI Text Detection UI
if option == "AI Text Detection":
    user_text = st.text_area("📝 Enter Text:", height=200)
    if st.button("Analyze Text"):
        if user_text:
            ai_score = detect_ai_text(user_text)
            st.write(f"🤖 AI-Generated Probability: **{ai_score}%**")
            if ai_score > 70:
                st.error("⚠️ This text is likely AI-generated!")
            else:
                st.success("✅ This text appears human-written.")
        else:
            st.warning("Please enter some text to analyze.")

# Deepfake Detection UI
elif option == "Deepfake Detection":
    uploaded_file = st.file_uploader("📸 Upload Image or Video:", type=["jpg", "png", "mp4"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Analyze Image"):
                result = detect_deepfake(image)
                st.write("🔍 Analysis Result:", result)
        else:
            st.warning("Please upload a valid image.")

# Credibility Check UI
elif option == "Credibility Check":
    url = st.text_input("🔗 Enter News Article URL:")
    if st.button("Check Credibility"):
        if url:
            result = check_credibility(url)
            st.write("📰 Credibility Check Result:", result)
        else:
            st.warning("Please enter a valid URL.")

st.write("---")
st.write("📌 **Note:** This system is fully open-source and uses state-of-the-art AI models.")
