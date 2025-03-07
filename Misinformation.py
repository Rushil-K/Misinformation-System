import streamlit as st
import requests
import torch
import cv2
import numpy as np
from transformers import pipeline
from bs4 import BeautifulSoup
from PIL import Image
from torchvision import transforms

# ============================== #
#       Load NLP Fake News Model #
# ============================== #
st.title("AI-Powered Misinformation Detection System")

@st.cache_resource
def load_nlp_model():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fakenews")

nlp_model = load_nlp_model()

def predict_fake_news(text):
    result = nlp_model(text)
    return {"label": result[0]['label'], "confidence": result[0]['score']}

# ============================== #
#       Load Deepfake Model      #
# ============================== #
@st.cache_resource
def load_deepfake_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True)
    model.eval()
    return model

deepfake_model = load_deepfake_model()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def detect_deepfake(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = deepfake_model(image)
    confidence = torch.nn.functional.softmax(output, dim=1)[0]
    is_fake = confidence[1].item() > 0.5  # Threshold for fake detection
    return {"label": "FAKE" if is_fake else "REAL", "confidence": confidence[1].item()}

# ============================== #
#     Credibility Scoring API    #
# ============================== #
FACT_CHECK_API_KEY = "AIzaSyAFGhCD-sBz5RhmZ8v7I3iMomAKXApzHS8"  # Replace with a valid API Key

def check_fact_via_google_factcheck(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return [{"text": claim['text'], "rating": claim['claimReview'][0]['textualRating']} for claim in data['claims']]
    return [{"text": "No fact-check found", "rating": "Unknown"}]

def check_fact_via_web_scraping(query):
    search_url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    for link in soup.select(".BVG0Nb"):
        articles.append(link.get_text())

    return articles if articles else ["No credible source found."]

# ============================== #
#       Streamlit Dashboard      #
# ============================== #

# 1️⃣ **Fake News Detection (Text)**
st.subheader("Check Text for Fake News")
text_input = st.text_area("Enter news text here:")
if st.button("Analyze Text"):
    if text_input:
        result = predict_fake_news(text_input)
        st.write(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f})")
    else:
        st.write("Please enter text to analyze.")

# 2️⃣ **Deepfake Image Detection**
st.subheader("Upload Image for Deepfake Detection")
image_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if image_file and st.button("Analyze Image"):
    with open("temp.jpg", "wb") as f:
        f.write(image_file.read())
    result = detect_deepfake("temp.jpg")
    st.write(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f})")

# 3️⃣ **Real-Time Credibility Scoring**
st.subheader("Fact-Check News Headlines")
query = st.text_input("Enter a news headline:")
if st.button("Verify Credibility"):
    google_results = check_fact_via_google_factcheck(query)
    scraped_results = check_fact_via_web_scraping(query)

    st.write("🔎 **Fact-Checked Articles:**")
    for res in google_results:
        st.write(f"- {res['text']} **({res['rating']})**")

    st.write("🌍 **News Sources:**")
    for article in scraped_results:
        st.write(f"- {article}")

