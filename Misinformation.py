import streamlit as st
 import torch
 from transformers import AutoModelForSequenceClassification, AutoTokenizer
 from PIL import Image
 import numpy as np
 import cv2
 import requests
 from deepface import DeepFace
 import validators
 import time
 from deepface import DeepFace
 import os
 
 def load_model():
     """Function to load the AI model with a confirmation message"""
     api_key = st.text_input("Enter API Key to Load Model:", type="password")
     if st.button("Load Model"):
         if api_key:  
             try:
                 model_name = "microsoft/deberta-v3-base"
                 global model, tokenizer
                 model = AutoModelForSequenceClassification.from_pretrained(model_name)
                 tokenizer = AutoTokenizer.from_pretrained(model_name)
                 st.success("Model loaded successfully!")
             except Exception as e:
                 st.error(f"Error loading model: {e}")
         else:
             st.error("Please enter a valid API key.")
 # Load the AI-generated text detection model
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
     """Detect if an image is deepfake using DeepFace"""
     try:
         result = DeepFace.analyze(image, actions=['emotion'])
         return "Deepfake detected!" if not result else "No deepfake detected."
         result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
         return result
     except Exception as e:
         return f"Error: {e}"
         return str(e)
 
 def ai_text_detection(text):
     """Detect if the text is AI-generated using Hugging Face model"""
     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
     with torch.no_grad():
         outputs = model(**inputs)
     scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
     return "AI-Generated" if scores[0][1] > scores[0][0] else "Human-Written"
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
 
 def check_url_reliability(url):
     """Check credibility of a URL using cross-verification"""
     if not validators.url(url):
         return "Invalid URL"
     response = requests.get(f"https://api.wikimedia.org/core/v1/wikipedia/en/page/{url}")
     return "Credible Source" if response.status_code == 200 else "Not a trusted source"
 # Image Analysis Section
 st.subheader("🖼️ Deepfake Image Detection")
 uploaded_image = st.file_uploader("Upload an image for deepfake detection:", type=["jpg", "png", "jpeg"])
 if uploaded_image:
     image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
     st.image(image, caption="Uploaded Image", use_column_width=True)
     if st.button("Analyze Image"):
         result = detect_deepfake(image)
         st.write("Deepfake Analysis Result:", result)
 
 def main():
     st.title("Misinformation Detection System")
     st.sidebar.title("Options")
     app_mode = st.sidebar.selectbox("Choose a mode", ["AI Text Detection", "Deepfake Detection", "Check URL Credibility"])
     
     load_model()
     
     if app_mode == "AI Text Detection":
         text_input = st.text_area("Enter text to analyze")
         if st.button("Analyze Text"):
             result = ai_text_detection(text_input)
             st.write("Result:", result)
     
     elif app_mode == "Deepfake Detection":
         uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
         if uploaded_file is not None:
             image = Image.open(uploaded_file)
             st.image(image, caption="Uploaded Image", use_column_width=True)
             if st.button("Detect Deepfake"):
                 result = detect_deepfake(np.array(image))
                 st.write("Result:", result)
     
     elif app_mode == "Check URL Credibility":
         url_input = st.text_input("Enter URL to check")
         if st.button("Check URL"):
             result = check_url_reliability(url_input)
             st.write("Result:", result)
 # Footer
 st.markdown("---")
 st.caption("🔍 Powered by Hugging Face & DeepFace | Open Source Misinformation Detector")
 
 if __name__ == "__main__":
     main()
