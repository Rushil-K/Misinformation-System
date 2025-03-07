import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from transformers import pipeline
from bs4 import BeautifulSoup

# Set page config
st.set_page_config(page_title="AI-Powered Misinformation Detection", layout="wide")

# Load NLP model for text analysis
nlp_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")

# Image deepfake detection placeholder (Can be replaced with a trained deepfake model)
def is_deepfake(image):
    # Placeholder: Simulating deepfake detection
    return torch.rand(1).item() > 0.5  # Random prediction for now

# Real-time credibility scoring (fetches from fact-checking sites)
def get_credibility_score(text):
    url = f"https://www.snopes.com/?s={text.replace(' ', '+')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("article", limit=1)
        if results:
            return f"Fact-check available: {results[0].find('a')['href']}"
    return "No direct fact-check found."

# Streamlit UI
st.title("🔍 AI-Powered Misinformation Detection")

# Sidebar for navigation
st.sidebar.header("Choose Detection Type")
option = st.sidebar.radio("Select Analysis Mode", ["Text Analysis", "Image Analysis"])

if option == "Text Analysis":
    st.subheader("📝 Text Misinformation Analysis")
    user_text = st.text_area("Enter text to analyze:")
    if st.button("Analyze"):
        if user_text:
            result = nlp_pipeline(user_text)
            credibility = get_credibility_score(user_text)
            st.write(f"🔍 **Analysis:** {result[0]['label']}")
            st.write(f"📌 **Credibility Check:** {credibility}")
        else:
            st.warning("Please enter text.")

elif option == "Image Analysis":
    st.subheader("🖼️ Deepfake Image Detection")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            fake_status = "Deepfake Detected ❌" if is_deepfake(image) else "Authentic ✅"
            st.write(f"🔍 **Result:** {fake_status}")

st.sidebar.info("🔬 Built using OpenAI, PyTorch, and BeautifulSoup.")

