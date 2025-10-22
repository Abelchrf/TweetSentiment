import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    ds = load_dataset("tweet_eval", "sentiment")["test"]
    labels_names = ds.features["label"].names

    y_true = []
    y_pred = []
    for i in range(0, len(ds), 64):
        batch = ds[i : i + 64]
        inputs = tokenizer(
            batch["text"], truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"])

    print(classification_report(y_true, y_pred, target_names=labels_names))


if __name__ == "__main__":
    main()
