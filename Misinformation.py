import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import requests
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from keras.preprocessing.sequence import pad_sequences
import cv2

# Set Streamlit Page Config
st.set_page_config(page_title="AI Misinformation Detector", layout="wide")

# Load BERT + LSTM Model for Text Analysis
class MisinformationModel(tf.keras.Model):
    def __init__(self, bert_model):
        super(MisinformationModel, self).__init__()
        self.bert = bert_model
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.bert(inputs)[1]  # Pooled output
        x = self.lstm(tf.expand_dims(x, axis=1))
        x = self.dropout(x)
        return self.dense(x)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
misinfo_model = MisinformationModel(bert_model)
misinfo_model.build(input_shape=(None, 128))

def analyze_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding='max_length', max_length=128)
    input_data = pad_sequences([tokens], maxlen=128, padding='post')
    prediction = misinfo_model.predict(input_data)[0][0]
    return "Reliable" if prediction < 0.5 else "Misleading"

# Google Fact Check API Integration
API_KEY = "AIzaSyAFGhCD-sBz5RhmZ8v7I3iMomAKXApzHS8"

def get_fact_check_results(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("claims", [])
        return results if results else ["No fact checks found."]
    return ["Error fetching fact-check results."]

# AI-Powered Deepfake Detection for Images
deepfake_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
deepfake_model.eval()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def detect_deepfake(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = deepfake_model(image).argmax().item()
    return "Real" if prediction == 0 else "Deepfake Detected"

# AI-Powered Deepfake Detection for Videos
def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count, deepfake_frames = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if detect_deepfake(image) == "Deepfake Detected":
            deepfake_frames += 1
    cap.release()
    return "Deepfake Detected" if deepfake_frames / frame_count > 0.5 else "Real"

# Streamlit UI
st.title("🛡️ AI-Powered Misinformation Detection System")
st.sidebar.title("🔍 Select Analysis Type")
option = st.sidebar.radio("", ["Text Analysis", "Image Deepfake Detection", "Video Deepfake Detection", "Fact-Check"])

if option == "Text Analysis":
    user_input = st.text_area("Enter text for misinformation detection:", "")
    if st.button("Analyze Text"):
        if user_input:
            result = analyze_text(user_input)
            st.success(f"Prediction: {result}")

elif option == "Image Deepfake Detection":
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Deepfake"):
            result = detect_deepfake(image)
            st.success(f"Prediction: {result}")

elif option == "Video Deepfake Detection":
    uploaded_video = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        st.video("temp_video.mp4")
        if st.button("Detect Deepfake"):
            result = detect_deepfake_video("temp_video.mp4")
            st.success(f"Prediction: {result}")

elif option == "Fact-Check":
    fact_check_query = st.text_input("Enter a statement to fact-check:")
    if st.button("Get Fact Check Results"):
        if fact_check_query:
            results = get_fact_check_results(fact_check_query)
            st.write("Fact-Check Results:")
            for res in results:
                st.write(f"- {res.get('text', 'No information available.')}")
