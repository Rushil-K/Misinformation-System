import streamlit as st
import os
import requests
import numpy as np
import cv2
import torch
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from keras.models import load_model
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

# Load BERT Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load LSTM model (ensure you have a pre-trained model)
lstm_model = load_model("lstm_model.h5")  # Replace with your LSTM model path

# Initialize Streamlit UI
st.set_page_config(page_title="Misinformation Detector", layout="wide")

st.title("🛡️ AI-Powered Misinformation Detector")
st.write("Analyze text, images, and videos for potential misinformation.")

# Sidebar options
option = st.sidebar.radio("Choose Analysis Type:", ["Text Verification", "Image Deepfake Detection", "Video Deepfake Detection"])

# --- Function: Google Fact Check API ---
def check_fact(text):
    service = build("factchecktools", "v1alpha1", developerKey=FACT_CHECK_API_KEY)
    request = service.claims().search(query=text, languageCode="en")
    response = request.execute()

    if "claims" in response:
        claim = response["claims"][0]
        result = {
            "text": claim["text"],
            "claimant": claim.get("claimant", "Unknown"),
            "rating": claim["claimReview"][0]["textualRating"],
            "source": claim["claimReview"][0]["publisher"]["name"],
        }
        return result
    else:
        return None

# --- Function: BERT & LSTM Text Analysis ---
def analyze_text(text):
    # Tokenize input for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    bert_output = bert_model(**inputs)
    bert_score = torch.sigmoid(bert_output.logits).item()

    # Predict using LSTM model
    lstm_prediction = lstm_model.predict(np.array([text]))[0]
    lstm_score = float(lstm_prediction[0])

    # Final score (weighted average)
    final_score = (bert_score * 0.6) + (lstm_score * 0.4)
    return final_score

# --- Function: Deepfake Detection (Image) ---
def detect_deepfake_image(image):
    # Load pre-trained deepfake detection model (Replace with actual model)
    deepfake_model = load_model("deepfake_model.h5")  

    # Convert image for prediction
    img_array = cv2.resize(image, (224, 224)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = deepfake_model.predict(img_array)[0][0]
    return prediction  # Higher value = more likely deepfake

# --- Function: Deepfake Detection (Video) ---
def detect_deepfake_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    deepfake_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_score = detect_deepfake_image(frame)  # Reusing image deepfake function
        deepfake_scores.append(frame_score)
        frame_count += 1

        if frame_count > 100:  # Limit analysis to 100 frames for efficiency
            break

    cap.release()
    avg_score = np.mean(deepfake_scores)
    return avg_score

# --- UI: Text Verification ---
if option == "Text Verification":
    text_input = st.text_area("Enter text to verify:", "")
    if st.button("Analyze Text"):
        if text_input.strip():
            # Fact-check API
            fact_result = check_fact(text_input)

            # AI Model Analysis
            ai_score = analyze_text(text_input)

            st.subheader("🔍 AI Analysis")
            st.write(f"Likelihood of misinformation: **{ai_score:.2%}**")

            if fact_result:
                st.subheader("✅ Google Fact Check Result")
                st.write(f"Claim: **{fact_result['text']}**")
                st.write(f"Claimant: {fact_result['claimant']}")
                st.write(f"Rating: **{fact_result['rating']}**")
                st.write(f"Source: **{fact_result['source']}**")
            else:
                st.warning("No fact-check results found. Cross-verify manually.")
        else:
            st.error("Please enter text to analyze.")

# --- UI: Image Deepfake Detection ---
elif option == "Image Deepfake Detection":
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            deepfake_score = detect_deepfake_image(image)
            st.write(f"Deepfake Likelihood: **{deepfake_score:.2%}**")

# --- UI: Video Deepfake Detection ---
elif option == "Video Deepfake Detection":
    uploaded_video = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Analyze Video"):
            deepfake_score = detect_deepfake_video(uploaded_video)
            st.write(f"Deepfake Likelihood: **{deepfake_score:.2%}**")

# Footer
st.markdown("---")
st.markdown("🔍 Built with **BERT, LSTM, Google Fact Check API, and Deepfake Detection Models**")
