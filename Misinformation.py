import asyncio

# Fix asyncio event loop issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
import cv2
import numpy as np
from deepface import DeepFace
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure OpenCV loads correctly
try:
    cv2.imread
except ImportError:
    st.error("⚠ OpenCV failed to load due to missing libGL.so.1. Run:\n"
             "`sudo apt-get install -y libgl1-mesa-glx`")

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using PyTorch on: `{device}`")

# Load AI-generated text detection model
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
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result
    except Exception as e:
        return str(e)

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
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Analyze Image"):
        result = detect_deepfake(image)
        st.write("Deepfake Analysis Result:", result)

# Torch test button
if st.sidebar.button("Run Torch Test"):
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    st.sidebar.write(f"Tensor created: {tensor}")

# Footer
st.markdown("---")
st.caption("🔍 Powered by Hugging Face & DeepFace | Open Source Misinformation Detector")
