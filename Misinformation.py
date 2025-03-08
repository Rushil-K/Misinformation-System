import os
import streamlit as st
import requests
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load pre-trained LSTM model (assume it's saved as 'lstm_model.h5')
@st.cache_resource
def load_misinformation_model():
    return tf.keras.models.load_model("lstm_model.h5")

model = load_misinformation_model()

# Function to predict misinformation using NLP
def predict_misinformation(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding="max_length", max_length=512)
    tokens = pad_sequences([tokens], maxlen=512, padding="post")
    prediction = model.predict(tokens)[0]
    return "Misleading" if prediction > 0.5 else "Reliable"

# Function to check facts using Google Fact Check API
def fact_check(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"
    response = requests.get(url).json()
    if "claims" in response:
        return response["claims"][0]["claimReview"][0]["textualRating"]
    return "No fact-check available"

# Function to detect deepfakes in an image
def detect_deepfake(image):
    # Placeholder: Replace with a trained deepfake detection model
    return "Likely Real"

# Streamlit UI
st.title("Misinformation Detection System")

st.sidebar.header("Choose Input Type")
option = st.sidebar.selectbox("Select", ["Text Analysis", "Fact Check", "Deepfake Detection"])

if option == "Text Analysis":
    user_text = st.text_area("Enter text to analyze for misinformation:")
    if st.button("Analyze"):
        result = predict_misinformation(user_text)
        st.write(f"Prediction: **{result}**")

elif option == "Fact Check":
    query = st.text_input("Enter a claim to fact-check:")
    if st.button("Check"):
        result = fact_check(query)
        st.write(f"Fact Check Result: **{result}**")

elif option == "Deepfake Detection":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        result = detect_deepfake(image)
        st.write(f"Deepfake Detection Result: **{result}**")
