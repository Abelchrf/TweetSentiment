from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def load_tweet_eval(tokenizer_name: str = "distilbert-base-uncased"):
    """
    Charge le dataset tweet_eval/sentiment et prépare la tokenisation + collator.
    Renvoie: (dataset_tokenized, tokenizer, data_collator, label_names)
    """
    ds = load_dataset("tweet_eval", "sentiment")  # splits: train/validation/test
    label_names = ds["train"].features["label"].names  # ['negative','neutral','positive']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    ds_tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return ds_tokenized, tokenizer, data_collator, label_names
