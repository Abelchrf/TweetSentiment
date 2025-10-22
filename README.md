# TweetSentiment — DistilBERT Fine-Tuning

Fine-tuning de `distilbert-base-uncased` pour la **classification de sentiments** sur des tweets.

## ✨ Points clés
- **Dataset** : `tweet_eval` / `sentiment` (3 classes : negative / neutral / positive)
- **Modèle** : DistilBERT (Transformers)
- **Métriques** : Accuracy, F1 macro
- **Démo** : Streamlit (saisie d’un tweet → prédiction)

## 🔧 Installation
```bash
python -m venv .venv && source .venv/bin/activate  # Win: .venv\Scripts\activate
pip install -r requirements.txt
