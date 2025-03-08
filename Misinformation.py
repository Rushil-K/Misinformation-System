import streamlit as st
import re
from transformers import pipeline
from textblob import TextBlob
from datetime import datetime
import torch

# Streamlit app setup
st.set_page_config(page_title="Misinformation & AI Text Detection", page_icon="🧐", layout="wide")
st.title("Misinformation and AI Text Detection")
st.markdown("""
This app detects misinformation and identifies AI-generated text using open-source models and NLP tools.
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

# Load GPT-2 Output Detector for AI Detection
ai_detector = pipeline("text-classification", model="huggingface/gpt2-output-detector", device=0 if torch.cuda.is_available() else -1)

# Function to check misinformation
def check_misinformation(text):
    result = misinformation_model(text)
    label = result[0]['label']
    confidence = result[0]['score']
    return label, confidence

# Function to check AI-generated text
def check_ai_generated(text):
    result = ai_detector(text)
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
                
                # Check for AI-generated text
                ai_label, ai_confidence = check_ai_generated(cleaned_text)
                
                # Check for misinformation
                misinformation_label, misinformation_confidence = check_misinformation(cleaned_text)
                
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
