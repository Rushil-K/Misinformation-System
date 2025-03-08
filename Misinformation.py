import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime
import torch

# Load environment variables (if needed)
load_dotenv()

# Streamlit app setup
st.set_page_config(page_title="Misinformation Detection", page_icon="🧐", layout="wide")
st.title("Misinformation Detection with Open Source Models")
st.markdown("""
This app detects misinformation and performs sentiment analysis using open-source models and NLP tools.
""")

# Sidebar for analysis options
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

# Load Misinformation Detection Model from Hugging Face
misinformation_model = pipeline("text-classification", model="roberta-base-openai-detector", device=0 if torch.cuda.is_available() else -1)

# Function to check misinformation
def check_misinformation(text):
    result = misinformation_model(text)
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
                
                # Check for misinformation
                label, confidence = check_misinformation(cleaned_text)
                
                # Analyze sentiment and credibility score
                score, sentiment = analyze_credibility(cleaned_text)
                
                # Display results
                st.write(f"**Misinformation Detection Result**: {label} (Confidence: {confidence:.2f})")
                st.write(f"**Credibility Score**: {score}%")
                st.write(f"**Sentiment**: {sentiment:.2f} (-1 negative, 0 neutral, 1 positive)")

                # Provide alerts based on results
                if label == "LABEL_1" and confidence > 0.75:  # 'LABEL_1' indicates misinformation
                    st.error("⚠️ This content is likely misinformation!")
                else:
                    st.success("✅ This content appears credible.")

    # Footer
    st.write("---")
    st.write("Note: This app uses an open-source pre-trained model for misinformation detection.")
    st.write(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

# Run the app
if __name__ == "__main__":
    main()
