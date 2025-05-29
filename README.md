

# 🚀 NLP Model Fine-Tuning, Text Classification, and Text Generation

This repository demonstrates multiple Natural Language Processing (NLP) tasks using Hugging Face Transformers. It includes:
- Fine-Tuning a sentiment analysis model (siebert/sentiment-roberta-large-English).
- Text Classification using (facebook/bart-large-mnli, nlptown/bert-base-multilingual-uncased-sentiment, cardiffnlp/twitter-roberta-base-sentiment).
- Text Generation using (facebook/bart-large-cnn).
Each task has been implemented using Google Colab, and the respective notebook links are included.

## 🛠️ 1. Fine-Tuning Sentiment RoBERTa Large (English)
Model Used:
- siebert/sentiment-roberta-large-English – A RoBERTa-based sentiment analysis model.
Dataset:
- nvidia/OpenCodeReasoning – Used for fine-tuning.
Fine-Tuning Process:
- Model Loading:
- The sentiment-roberta-large-English model was loaded from Hugging Face.
- Dataset Preprocessing:
- Tokenization of the dataset using the model’s tokenizer.
- Formatting the dataset into Tensor format for training.
- Training:
- Fine-tuned using  Google Colab’s GPU.
- Used AdamW optimizer and a suitable learning rate.
- Set training epochs and batch size for efficient training.
- Logged training loss and monitored model improvements.
- Evaluation:
- Validated the fine-tuned model on test data.
- Generated accuracy and loss metrics for evaluation.
🔗 Colab Notebook: https://colab.research.google.com/drive/11cZyqShYyryehUhWXbkjvNYU7MaTxWR4?usp=drive_link

## 🏷️ 2. Text Classification
Models Used:
- facebook/bart-large-mnli – Zero-shot classification based on natural language inference.
- nlptown/bert-base-multilingual-uncased-sentiment – Multilingual sentiment analysis.
- cardiffnlp/twitter-roberta-base-sentiment – Sentiment classification specialized for Twitter data.
Implementation Steps:
- Model Selection:
- Used Hugging Face pipeline to load models.
- Preprocessing:
- Tokenized input text and prepared batches for prediction.
- Prediction:
- Performed zero-shot classification with facebook/bart-large-mnli.
- Conducted sentiment analysis with BERT-based models.
Example Usage:
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("I love this movie!", candidate_labels=["positive", "negative", "neutral"])
print(result)


🔗 Colab Notebook: https://colab.research.google.com/drive/1erBBnPC3G1oErvunXLuB6bzMU3Ra9RGT?usp=drive_link

## ✍️ 3. Text Generation
Model Used:
- facebook/bart-large-cnn – Used for summarization.
Implementation Steps:
- Model Loading:
- Loaded facebook/bart-large-cnn from Hugging Face.
- Text Preprocessing:
- Formatted input text properly for summarization.
- Text Generation:
- Generated a summarized output using the model.
Example Usage:
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer("Your long article or paragraph goes here.", max_length=50, min_length=20, do_sample=False)
print(summary)


🔗 Colab Notebook: https://colab.research.google.com/drive/1erBBnPC3G1oErvunXLuB6bzMU3Ra9RGT?usp=drive_link

### 🔧 Dependencies
Before running the notebooks, install the necessary dependencies:
pip install transformers datasets torch hugging face



### 📌 Credits
- Base Models:
- siebert/sentiment-roberta-large-English
- facebook/bart-large-mnli
- nlptown/bert-base-multilingual-uncased-sentiment
- cardiffnlp/twitter-roberta-base-sentiment
- facebook/bart-large-cnn
- Dataset: Nvidia/OpenCodeReasoning
- Framework: Hugging Face Transformers


