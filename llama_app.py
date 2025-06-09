import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "finetuned-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = {0: "Negative😒", 1: "Positive😊"}  

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        return id2label[predicted_class_id]

st.set_page_config(page_title="Sentiment Analyzer", layout="centered", page_icon="💬")
st.title("Sentiment Analyzer")
st.markdown("Enter any text and get the predicted sentiment using your fine-tuned model.")

text_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        label = predict_sentiment(text_input)
        st.success(f"Predicted Sentiment: **{label}**")
    else:
        st.warning("Please enter some text to analyze.")
