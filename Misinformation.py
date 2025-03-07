import streamlit as st
import asyncio
from transformers import pipeline

# Fix asyncio RuntimeError: no running event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Load NLP model with error handling
@st.cache_resource(show_spinner=True)
def load_nlp_model():
    try:
        model_name = "mrm8488/bert-tiny-finetuned-fakenews"
        return pipeline("text-classification", model=model_name)
    except Exception as e:
        st.warning(f"Error loading model '{model_name}': {e}. Switching to an alternative model.")
        return pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Load the model
nlp_model = load_nlp_model()

# Streamlit App UI
st.set_page_config(page_title="Misinformation Detector", page_icon="📰", layout="wide")

st.title("📰 AI-Powered Misinformation Detection System")
st.write("Enter a news headline or text to check if it's fake or real.")

# User Input
user_input = st.text_area("🔍 Enter News Text:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            prediction = nlp_model(user_input)[0]
            label = prediction["label"]
            confidence = prediction["score"]

            # Display result
            st.subheader("🧐 Analysis Result")
            st.write(f"**Prediction:** `{label}`")
            st.write(f"**Confidence Score:** `{confidence:.4f}`")

            # Interpretation
            if "fake" in label.lower():
                st.error("⚠️ This text is likely misinformation or fake news.")
            else:
                st.success("✅ This text appears to be real or credible.")

# Footer
st.markdown("---")
st.markdown("🔬 Built with [Hugging Face Transformers](https://huggingface.co) | 🚀 Streamlit-Powered AI System")

