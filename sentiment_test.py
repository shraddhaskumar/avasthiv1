import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load model and tokenizer
roberta_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model=roberta_model, tokenizer=roberta_tokenizer)

def analyze_sentiment(text):
    """Predict sentiment using nlptown/bert-base-multilingual-uncased-sentiment"""
    result = sentiment_pipeline(text)[0]
    sentiment_map = {
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive"
    }
    return sentiment_map.get(result["label"], "neutral")

# Load CSV file
df = pd.read_csv("testdata.csv")

# Apply sentiment analysis
df["predicted_sentiment"] = df["query"].apply(analyze_sentiment)

# Compare predicted vs actual sentiment
df["correct"] = df["predicted_sentiment"] == df["actual_sentiment"]

# Calculate Accuracy
accuracy = df["correct"].mean() * 100

# Convert categorical labels to numeric for metric calculation
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["actual_sentiment_num"] = df["actual_sentiment"].map(label_mapping)
df["predicted_sentiment_num"] = df["predicted_sentiment"].map(label_mapping)

# Compute additional metrics
precision = precision_score(df["actual_sentiment_num"], df["predicted_sentiment_num"], average="macro")
recall = recall_score(df["actual_sentiment_num"], df["predicted_sentiment_num"], average="macro")
f1 = f1_score(df["actual_sentiment_num"], df["predicted_sentiment_num"], average="macro")
conf_matrix = confusion_matrix(df["actual_sentiment_num"], df["predicted_sentiment_num"])

# Print results
print(f"Sentiment Analysis Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Save results
df.to_csv("sentiment_analysis_results.csv", index=False)
