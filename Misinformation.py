import streamlit as st
import pandas as pd
import re
import aiohttp
import asyncio
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime
from googleapiclient.discovery import build
import os

# Load environment variables
load_dotenv()

# Google Fact Check API credentials
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")
fact_check_service = build("factchecktools", "v1alpha1", developerKey=FACT_CHECK_API_KEY)

# Streamlit app setup
st.set_page_config(page_title="Misinformation Detection", page_icon="🧐", layout="wide")
st.title("Misinformation Detection with Google Fact-Check")
st.markdown("""
This app allows you to analyze claims for misinformation by cross-referencing them using the [Google Fact Check Tools](https://developers.google.com/fact-check/tools).
""")
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Text", "URL"])

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#', '', text)
    return text.strip()

# Function to analyze credibility
def analyze_credibility(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    credibility_score = 100

    # Basic checks to add more credibility criteria
    if text.isupper() or "!!!" in text:
        credibility_score -= 20
    if any(word in text.lower() for word in ["always", "never", "every", "all"]):
        credibility_score -= 15
    if len(text.split()) < 10:
        credibility_score -= 10  # Short posts might lack context
    
    return credibility_score, sentiment

# Async function to check facts using Google Fact Check API
async def check_facts_async(query):
    async with aiohttp.ClientSession() as session:
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
        async with session.get(url) as response:
            response_data = await response.json()
            if 'claims' in response_data:
                claims = response_data['claims']
                if claims:
                    return claims[0]['textualRating'], claims[0]['claimReviewDate']
            return "No fact check found", None

# Main async function to run the analysis
async def main():
    # Text Analysis
    if analysis_type == "Text":
        text_input = st.text_area("Enter text to analyze", height=200)
        if st.button("Analyze Text"):
            if text_input:
                cleaned_text = clean_text(text_input)
                score, sentiment = analyze_credibility(cleaned_text)
                st.write(f"**Credibility Score**: {score}%")
                st.write(f"**Sentiment**: {sentiment:.2f} (-1 negative, 0 neutral, 1 positive)")

                if score < 50:
                    st.error("⚠️ High likelihood of misinformation")
                elif score < 75:
                    st.warning("🔶 Possible misinformation detected")
                else:
                    st.success("✅ Content appears credible")

                # Fact-checking using Google Fact Check API
                fact_check_result, fact_check_date = await check_facts_async(cleaned_text)
                st.write(f"**Fact Check**: {fact_check_result}")
                if fact_check_date:
                    st.write(f"**Fact Check Date**: {fact_check_date}")

    # URL Analysis (placeholder)
    elif analysis_type == "URL":
        st.write("URL analysis not implemented in this demo. Would require external fact-checking API integration.")

    # Footer
    st.write("---")
    st.write("Note: This uses Google Fact-Check API for analysis. For production use:")
    st.write("- Add machine learning models")
    st.write("- Integrate additional fact-checking APIs")
    st.write("- Enhance credibility scoring")
    st.write(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

# Run the async function inside Streamlit
if __name__ == "__main__":
    asyncio.run(main())
