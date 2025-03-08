import streamlit as st
import tweepy
import pandas as pd
from textblob import TextBlob
import re
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from datetime import datetime
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# Twitter API credentials (you'll need to set these in a .env file)
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

# Google Fact Check API credentials
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")
fact_check_service = build("factchecktools", "v1alpha1", developerKey=FACT_CHECK_API_KEY)

# Initialize Twitter API client
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)

# Streamlit app setup
st.title("Misinformation Detection App")
st.write("Analyze social media content for potential misinformation")

# Sidebar
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio("Analysis Type", ["X Post", "Text", "URL"])

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
    
    # Basic checks
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

# Function to fetch tweets using the Twitter API
async def fetch_tweets_async(user_id, num_posts):
    tweets = client.get_users_tweets(
        id=user_id,
        max_results=num_posts,
        tweet_fields=["created_at", "public_metrics"]
    )
    return tweets

# Main async function to run the analysis
async def main():
    # X Post Analysis (Async Version)
    if analysis_type == "X Post":
        username = st.text_input("Enter X username (without @)")
        num_posts = st.slider("Number of posts to analyze", 1, 20, 10)

        if st.button("Analyze X Posts"):
            if username:
                try:
                    # Get user ID
                    user = client.get_user(username=username)
                    if user.data:
                        user_id = user.data.id

                        # Asynchronous tweet fetching
                        tweets = await fetch_tweets_async(user_id, num_posts)

                        if tweets.data:
                            results = []
                            for tweet in tweets.data:
                                cleaned_text = clean_text(tweet.text)
                                score, sentiment = analyze_credibility(cleaned_text)

                                # Asynchronous fact-checking
                                fact_check_result, fact_check_date = await check_facts_async(cleaned_text)

                                results.append({
                                    "Text": cleaned_text,
                                    "Score": score,
                                    "Sentiment": sentiment,
                                    "Date": tweet.created_at,
                                    "Likes": tweet.public_metrics["like_count"],
                                    "Fact Check": fact_check_result,
                                    "Fact Check Date": fact_check_date
                                })

                            # Display results
                            df = pd.DataFrame(results)
                            st.write("Analysis Results:")
                            st.dataframe(df)

                            # Summary
                            avg_score = df["Score"].mean()
                            if avg_score < 50:
                                st.error(f"Average Credibility Score: {avg_score:.2f}% - High likelihood of misinformation")
                            elif avg_score < 75:
                                st.warning(f"Average Credibility Score: {avg_score:.2f}% - Possible misinformation")
                            else:
                                st.success(f"Average Credibility Score: {avg_score:.2f}% - Content appears credible")
                        else:
                            st.error("No posts found for this user")
                    else:
                        st.error("User not found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a username")

    # Text Analysis
    elif analysis_type == "Text":
        text_input = st.text_area("Enter text to analyze", height=200)
        if st.button("Analyze Text"):
            if text_input:
                cleaned_text = clean_text(text_input)
                score, sentiment = analyze_credibility(cleaned_text)
                st.write(f"Credibility Score: {score}%")
                st.write(f"Sentiment: {sentiment:.2f} (-1 negative, 0 neutral, 1 positive)")

                if score < 50:
                    st.error("High likelihood of misinformation")
                elif score < 75:
                    st.warning("Possible misinformation detected")
                else:
                    st.success("Content appears credible")

    # URL Analysis (placeholder)
    elif analysis_type == "URL":
        st.write("URL analysis not implemented in this demo. Would require external fact-checking API integration.")

    # Footer
    st.write("---")
    st.write("Note: This uses Twitter API v2 with basic analysis. For production use:")
    st.write("- Add machine learning models")
    st.write("- Integrate fact-checking APIs")
    st.write("- Enhance credibility scoring")
    st.write(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

# Run the async function inside Streamlit
if __name__ == "__main__":
    asyncio.run(main())
