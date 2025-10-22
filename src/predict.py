import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]

def predict(model_dir: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(logits.argmax(dim=-1))
    return LABELS[pred_id]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text", required=True)
    args = p.parse_args()

    label = predict(args.model_dir, args.text)
    print(label)

if __name__ == "__main__":
    main()
