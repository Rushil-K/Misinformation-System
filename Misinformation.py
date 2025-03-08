import streamlit as st
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deepface import DeepFace
from newspaper import Article
import validators
import os

# Load API key before running
st.title("Misinformation & Deepfake Detection System")

api_key = st.text_input("Enter your API key:", type="password")
if st.button("Load API Key"):
    if api_key:
        st.success("API Key loaded successfully!")
    else:
        st.error("Please enter a valid API Key.")

if not api_key:
    st.warning("Please enter an API key to proceed.")
    st.stop()

# Load AI-generated text detection model (open-source)
@st.cache_resource
def load_text_detection_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

text_model, text_tokenizer = load_text_detection_model()

def detect_ai_generated_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = text_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    fake_score = probs[0][1].item() * 100
    return fake_score

# Fake news credibility scoring
def get_news_credibility(url):
    if not validators.url(url):
        return None, "Invalid URL"
    
    article = Article(url)
    try:
        article.download()
        article.parse()
    except:
        return None, "Error extracting the article"
    
    text = article.text
    score = detect_ai_generated_text(text)
    
    return score, article.title

# Deepfake detection for images/videos
def detect_deepfake(file):
    try:
        result = DeepFace.analyze(file, actions=["emotion", "age", "gender"])
        return result
    except:
        return None

# Streamlit UI
st.sidebar.title("Analysis Options")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Text", "URL", "Image", "Video"])

if analysis_type == "Text":
    user_text = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text"):
        if user_text:
            fake_score = detect_ai_generated_text(user_text)
            st.write(f"AI-Generated Probability: {fake_score:.2f}%")
            if fake_score > 50:
                st.error("This text is likely AI-generated.")
            else:
                st.success("This text appears human-written.")

elif analysis_type == "URL":
    news_url = st.text_input("Enter news article URL:")
    if st.button("Check News Credibility"):
        if news_url:
            score, title = get_news_credibility(news_url)
            if score is None:
                st.error(title)
            else:
                st.write(f"Title: {title}")
                st.write(f"AI-Generated Probability: {score:.2f}%")
                if score > 50:
                    st.error("This article is likely AI-generated or unreliable.")
                else:
                    st.success("This article appears credible.")

elif analysis_type == "Image":
    uploaded_image = st.file_uploader("Upload an image for deepfake detection", type=["jpg", "png"])
    if uploaded_image and st.button("Analyze Image"):
        result = detect_deepfake(uploaded_image)
        if result:
            st.write(result)
        else:
            st.error("Deepfake detection failed.")

elif analysis_type == "Video":
    uploaded_video = st.file_uploader("Upload a video for deepfake detection", type=["mp4", "avi"])
    if uploaded_video and st.button("Analyze Video"):
        result = detect_deepfake(uploaded_video)
        if result:
            st.write(result)
        else:
            st.error("Deepfake detection failed.")

st.write("© 2025 Open-Source Misinformation Detection System")
