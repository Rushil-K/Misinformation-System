import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import numpy as np
import requests
from deepface import DeepFace
import validators
import time
import os

# Load AI model for AI-generated text detection
@st.cache_resource
def load_text_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, text_model = load_text_model()

# Function to detect AI-generated text
def detect_ai_generated_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_score = scores[0][1].item()
    return ai_score

# Function for deepfake detection in images
def detect_deepfake(image):
    """Detect if an image is a deepfake using DeepFace"""
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result
    except Exception as e:
        return str(e)

# Function to check URL reliability
def check_url_reliability(url):
    """Check the credibility of a URL"""
    if not validators.url(url):
        return "Invalid URL"
    try:
        response = requests.get(url, timeout=5)
        return "✅ Credible Source" if response.status_code == 200 else "❌ Not a Trusted Source"
    except requests.RequestException:
        return "⚠ Could not verify URL credibility"

# Streamlit UI
st.set_page_config(page_title="Misinformation Detector", layout="wide")
st.title("🕵️ Misinformation Detection System")

# Sidebar for API key input
st.sidebar.header("🔑 Load Model API")
api_key = st.sidebar.text_input("Enter API Key (if required)", type="password")
if st.sidebar.button("Load API Key"):
    if api_key:
        st.sidebar.success("✅ API Key Loaded Successfully!")
    else:
        st.sidebar.warning("⚠ No API Key entered. Some features may not work.")

# Text Analysis Section
st.subheader("📝 AI-Generated Text Detection")
user_text = st.text_area("Enter text to analyze:")
if st.button("Analyze Text"):
    if user_text:
        ai_score = detect_ai_generated_text(user_text)
        st.write(f"🤖 AI-Generated Probability: {ai_score:.2%}")
        if ai_score > 0.5:
            st.error("🚨 This text is likely AI-generated!")
        else:
            st.success("✅ This text appears to be human-written.")
    else:
        st.warning("⚠ Please enter some text.")

# Image Analysis Section
st.subheader("🖼️ Deepfake Image Detection")
uploaded_image = st.file_uploader("Upload an image for deepfake detection:", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Analyze Image"):
        result = detect_deepfake(np.array(image))  # Convert PIL image to NumPy array
        st.write("Deepfake Analysis Result:", result)

# URL Credibility Check Section
st.subheader("🔗 URL Credibility Check")
url_input = st.text_input("Enter URL to check:")
if st.button("Check URL"):
    result = check_url_reliability(url_input)
    st.write("URL Credibility Result:", result)

# Footer
st.markdown("---")
st.caption("🔍 Powered by Hugging Face & DeepFace | Open Source Misinformation Detector")

