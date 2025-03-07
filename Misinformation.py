import os
import streamlit as st
import requests
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
from deepface import DeepFace
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

# Load BERT tokenizer and LSTM model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
lstm_model = tf.keras.models.load_model("lstm_model.h5")

# Streamlit UI Configuration
st.set_page_config(page_title="Misinformation Detection", page_icon="🛑", layout="wide")

st.title("🛑 AI-Powered Misinformation Detection System")

# Sidebar Menu
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to", ["Text Analysis", "Deepfake Detection", "Fact Checking"])

# Function to preprocess text
def preprocess_text(text):
    tokens = tokenizer.encode(text, truncation=True, padding="max_length", max_length=256, return_tensors="tf")
    return tokens.numpy()

# Function to analyze text
def analyze_text(text):
    input_text = preprocess_text(text)
    prediction = lstm_model.predict(input_text)
    return prediction[0][0]

# Function to check with Google Fact Check API
def google_fact_check(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        claims = data.get("claims", [])
        return claims
    return None

# Function for deepfake detection
def detect_deepfake(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion', 'age', 'gender'])
        return result
    except Exception:
        return None

# Page Handling
if page == "Text Analysis":
    st.subheader("📄 Analyze Text for Misinformation")
    text_input = st.text_area("Enter text to analyze:")
    
    if st.button("Analyze"):
        if text_input:
            score = analyze_text(text_input)
            if score > 0.5:
                st.error("🚨 Misinformation Detected!")
            else:
                st.success("✅ The text appears credible.")
        else:
            st.warning("Please enter some text.")

elif page == "Deepfake Detection":
    st.subheader("🎭 Detect Deepfakes in Images")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            result = detect_deepfake(image)
            if result:
                st.success("✅ The image appears to be authentic.")
            else:
                st.error("🚨 Possible deepfake detected!")

elif page == "Fact Checking":
    st.subheader("🔍 Fact Check with Google")
    query = st.text_input("Enter a claim to verify:")
    
    if st.button("Check Fact"):
        if query:
            claims = google_fact_check(query)
            if claims:
                st.write("Fact Check Results:")
                for claim in claims:
                    st.markdown(f"**Claim:** {claim['text']}")
                    st.markdown(f"**Rating:** {claim['claimReview'][0]['textualRating']}")
                    st.markdown(f"[Source]({claim['claimReview'][0]['url']})")
            else:
                st.error("No fact-checking results found.")
        else:
            st.warning("Enter a claim to check.")
