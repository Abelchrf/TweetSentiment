# src/train.py
import argparse
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
from src.data import load_tweet_eval


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--output_dir", default="models/distilbert-tweets")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    ds, tokenizer, data_collator, label_names = load_tweet_eval(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_names)
    )

    # --- Compatibilité large: on essaie d'abord la signature récente,
    # --- sinon on tombe sur une config minimale sans 'evaluation_strategy'.
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",   # <- peut être rejeté selon la version
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to=["none"],
        )
    except TypeError:
        # Fallback compatible (pas d'éval automatique pendant train)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=50,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],   # même si pas d'éval auto, on pourra appeler evaluate()
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Si la version ne faisait pas d'éval pendant le train, on force une éval à la fin :
    try:
        trainer.evaluate()
    except Exception:
        pass

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Modèle sauvegardé dans {args.output_dir}")


if __name__ == "__main__":
    main()
