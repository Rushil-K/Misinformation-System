import streamlit as st
from transformers import pipeline
from datetime import datetime
import re
from textblob import TextBlob
import os

# Streamlit app setup
st.set_page_config(page_title="Misinformation & AI Text Detection", page_icon="🧐", layout="wide")
st.title("Misinformation and AI Text Detection")
st.markdown("""
This app detects misinformation and identifies AI-generated text using open-source models and NLP tools.
""")

# Sidebar for API Key input
st.sidebar.header("API Key Configuration")
api_key = st.sidebar.text_input("Enter your API Key (for AI Detection Model)", type="password")

# Check if the API Key is provided
if not api_key:
    st.warning("Please enter a valid API Key to proceed.")
    st.stop()

# Option to select analysis type
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Text"])

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions and hashtags
    return text.strip()

# Function to analyze credibility and sentiment
def analyze_credibility(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    credibility_score = 100
    
    # Basic checks
    if text.isupper() or "!!!" in text:
        credibility_score -= 20
    if any(word in text.lower() for word in ["always", "never", "every", "all"]):
        credibility_score -= 15
    if len(text.split()) < 10:
        credibility_score -= 10  # Short posts might lack context
    
    return credibility_score, sentiment

# Load GPT-2 Output Detector (for AI-generated text detection)
@st.cache_resource
def load_ai_detector(api_key):
    # Here, you can integrate an API key for loading a model or use a local model.
    # For simplicity, we use a pre-trained model from Hugging Face
    ai_detector = pipeline("text-classification", model="openai-gpt2-output-detector", device=0 if torch.cuda.is_available() else -1)
    return ai_detector

# Load Fake News Detection Model (RoBERTa or a BERT-based model)
@st.cache_resource
def load_misinformation_model(api_key):
    # Here, you can integrate an API key for loading a model or use a local model.
    misinformation_model = pipeline("text-classification", model="roberta-base-openai-detector", device=0 if torch.cuda.is_available() else -1)
    return misinformation_model

# Function to check misinformation
def check_misinformation(text, model):
    result = model(text)
    label = result[0]['label']
    confidence = result[0]['score']
    return label, confidence

# Function to check AI-generated text
def check_ai_generated(text, model):
    result = model(text)
    label = result[0]['label']
    confidence = result[0]['score']
    return label, confidence

# Main function to run the analysis
def main():
    if analysis_type == "Text":
        text_input = st.text_area("Enter text to analyze", height=200)
        
        if st.button("Analyze Text"):
            if text_input:
                cleaned_text = clean_text(text_input)
                
                # Load models for analysis
                ai_detector = load_ai_detector(api_key)
                misinformation_model = load_misinformation_model(api_key)
                
                # Check for AI-generated text
                ai_label, ai_confidence = check_ai_generated(cleaned_text, ai_detector)
                
                # Check for misinformation
                misinformation_label, misinformation_confidence = check_misinformation(cleaned_text, misinformation_model)
                
                # Analyze sentiment and credibility score
                score, sentiment = analyze_credibility(cleaned_text)
                
                # Display results
                st.write(f"**AI Detection Result**: {ai_label} (Confidence: {ai_confidence:.2f})")
                st.write(f"**Misinformation Detection Result**: {misinformation_label} (Confidence: {misinformation_confidence:.2f})")
                st.write(f"**Credibility Score**: {score}%")
                st.write(f"**Sentiment**: {sentiment:.2f} (-1 negative, 0 neutral, 1 positive)")

                # Provide alerts based on results
                if ai_label == "LABEL_1" and ai_confidence > 0.75:  # 'LABEL_1' indicates AI-generated text
                    st.warning("⚠️ This content appears to be AI-generated!")
                elif misinformation_label == "LABEL_1" and misinformation_confidence > 0.75:  # Misinformation label
                    st.error("⚠️ This content is likely misinformation!")
                else:
                    st.success("✅ This content appears credible.")

    # Footer
    st.write("---")
    st.write("Note: This app uses open-source pre-trained models for misinformation and AI text detection.")
    st.write(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

# Run the app
if __name__ == "__main__":
    main()
