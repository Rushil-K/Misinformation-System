import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import lime.lime_text
import requests
import pandas as pd
import os
# -------------------- Load Model & Tokenizer --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "hybrid_bert_lstm.pth"

class HybridBERTLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=6):
        super(HybridBERTLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        return self.fc(lstm_out[:, -1, :])  # Last LSTM output

@st.cache_resource
def load_model():
    model = HybridBERTLSTM().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        url = "https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO/raw/main/hybrid_bert_lstm.pth"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')

model = load_model()
tokenizer = load_tokenizer()

# -------------------- Data Preprocessing --------------------
LABELS = ["False", "Half-True", "Mostly-True", "True", "Barely-True", "Pants-on-Fire"]

def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return LABELS[torch.argmax(outputs, dim=1).item()]

# -------------------- LIME Explainability --------------------
def lime_explanation(text):
    explainer = lime.lime_text.LimeTextExplainer(class_names=LABELS)

    def predictor(texts):
        return np.array([model(**tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(DEVICE))[0].cpu().numpy() for t in texts])

    exp = explainer.explain_instance(text, predictor, num_features=10)
    return exp.as_html()

# -------------------- Streamlit UI --------------------
st.title("📰 AI-Powered Fake News Detection")
st.write("🚀 Analyzing statements for misinformation using **Hybrid BERT + LSTM**.")

tab1, tab2 = st.tabs(["🔍 News Analysis", "📊 Explainability"])

with tab1:
    st.header("📜 Enter a News Statement")
    text_input = st.text_area("Paste your news statement here:")
    if st.button("Analyze News"):
        if text_input:
            prediction = predict(text_input)
            st.success(f"🛑 Prediction: **{prediction}**")
        else:
            st.warning("⚠️ Please enter a valid statement.")

with tab2:
    st.header("📊 LIME Explainability")
    text_explain = st.text_area("Enter text to explain model decision:")
    if st.button("Generate Explanation"):
        if text_explain:
            explanation = lime_explanation(text_explain)
            st.components.v1.html(explanation, height=600)
        else:
            st.warning("⚠️ Please enter a valid text.")

st.write("💡 Created with ❤️ by [Your Name]")
