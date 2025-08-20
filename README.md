# Sentiment Analysis on Amazon Reviews

This project demonstrates multiple approaches for sentiment analysis on Amazon product reviews: **basic Python sentiment analysis**, **Bidirectional LSTM neural network**, and **pretrained BERT model (Hugging Face)**. Users can interactively input sentences for predictions and evaluate model performance.

## Dataset

- **Source:** [Amazon Reviews Dataset](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv)  
- **Columns Used:** `reviewText` (text), `Positive` (1 = positive, 0 = negative)  
- **Size:** 20,000 reviews  

## Approaches

### Python Basic Sentiment Analysis (VADER)
- Uses `nltk`’s VADER for rule-based sentiment scoring.  
- Provides quick sentiment labels without training.  
- Good for baseline evaluation.  

### Neural Network (Bidirectional LSTM)
- Trains on the dataset for binary sentiment classification.  
- Preprocessing: lowercasing, removing numbers/punctuation, tokenization, padding.  
- Uses embedding layer, Bidirectional LSTM, and Dense output with sigmoid activation.  

### Pretrained BERT (Hugging Face)
- Uses `nlptown/bert-base-multilingual-uncased-sentiment` for sentence-level predictions.  
- Supports interactive input with immediate sentiment output.  
- Can output binary label, reasoning, or numeric rating depending on prompt.  

## Prompt Variations for BERT/LLM
1. `"Classify the sentiment of this review as Positive or Negative: {review}"` – binary label.  
2. `"Read the review and say if the user feels happy or unhappy: {review}"` – includes short explanation.  
3. `"Rate the sentiment 1 (very negative) to 5 (very positive): {review}"` – numeric score for fine-grained analysis.  

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1 Score.  
- Evaluation can be done interactively (user inputs sentences) or on a labeled dataset.  

## Troubleshooting
- **Overfitting:** Model performs well on training data but poorly on new data.  
- **Fixes:** Increase dataset, use regularization, simpler models, shuffle or augment data.  
