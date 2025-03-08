import streamlit as st
from transformers import pipeline
from textblob import TextBlob
import re
from datetime import datetime
import torch

# Streamlit app setup
st.set_page_config(page_title="Misinformation & AI Text Detection", page_icon="🧐", layout="wide")
st.title("Misinformation and AI Text Detection")
st.markdown("""
This app detects misinformation and identifies AI-generated text using open-source models and NLP tools.
""")

# Sidebar Configuration
st.sidebar.header("Text Analysis Options")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Text"])

# Function to clean the input text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions and hashtags
    return text.strip()

# Function to analyze sentiment and credibility
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

# Function to load models for AI detection and misinformation detection
@st.cache_resource
def load_models():
    # Load pre-trained models for misinformation detection and AI text detection
    misinformation_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    ai_detector_model = pipeline("text-classification", model="roberta-base-openai-detector")
    return misinformation_model, ai_detector_model

# Function to check for misinformation
def check_misinformation(text, model):
    candidate_labels = ["misinformation", "truth"]
    result = model(text, candidate_labels)
    label = result['labels'][0]
    score = result['scores'][0]
    return label, score

# Function to check if text is AI-generated
def check_ai_generated(text, model):
    result = model(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

# Main Function to run the analysis
def main():
    if analysis_type == "Text":
        text_input = st.text_area("Enter text to analyze", height=200)

        if st.button("Analyze Text"):
            if text_input:
                # Clean the input text
                cleaned_text = clean_text(text_input)

                # Load models
                misinformation_model, ai_detector_model = load_models()

                # Check for AI-generated text
                ai_label, ai_confidence = check_ai_generated(cleaned_text, ai_detector_model)

                # Check for misinformation
                misinformation_label, misinformation_confidence = check_misinformation(cleaned_text, misinformation_model)

                # Analyze sentiment and credibility
                credibility_score, sentiment = analyze_credibility(cleaned_text)

                # Display results
                st.write(f"**AI Detection Result**: {ai_label} (Confidence: {ai_confidence:.2f})")
                st.write(f"**Misinformation Detection Result**: {misinformation_label} (Confidence: {misinformation_confidence:.2f})")
                st.write(f"**Credibility Score**: {credibility_score}%")
                st.write(f"**Sentiment**: {sentiment:.2f} (-1 negative, 0 neutral, 1 positive)")

                # Provide alerts based on results
                if ai_label == "LABEL_1" and ai_confidence > 0.75:  # LABEL_1 indicates AI-generated text
                    st.warning("⚠️ This content appears to be AI-generated!")
                elif misinformation_label == "misinformation" and misinformation_confidence > 0.75:  # Misinformation label
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
