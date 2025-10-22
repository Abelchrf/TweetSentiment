import argparse
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def main(model_dir: str):
    st.set_page_config(page_title="TweetSentiment — DistilBERT")
    st.title("TweetSentiment — Démo DistilBERT")
    st.write("Tape un tweet et obtiens sa polarité (negative / neutral / positive).")

    tokenizer, model = load_model(model_dir)

    text = st.text_area("Tweet", height=120, placeholder="Ex: I absolutely love this product!")
    if st.button("Prédire"):
        if not text.strip():
            st.warning("Entre un texte.")
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_id = int(logits.argmax(dim=-1))
            st.subheader(f"Résultat : {LABELS[pred_id]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/distilbert-tweets")
    args, _ = parser.parse_known_args()
    main(args.model_dir)
