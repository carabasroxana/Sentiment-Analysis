# Sentiment Analysis: English and Romanian

This project implements a sentiment analysis pipeline for text in English and Romanian. It uses the NLTK library for English sentiment analysis and a transformer-based model for Romanian text. The application can identify the sentiment as **Positive**, **Negative**, or **Neutral** and dynamically updates evaluation metrics as new inputs are processed.

## Features

- **Language Detection**: Automatically detects whether the text is in English or Romanian using the `langdetect` library.
- **Text Preprocessing**: Cleans and tokenizes text, removes stopwords, and performs lemmatization.
- **Sentiment Analysis**:
  - For English: Uses NLTK's VADER sentiment analyzer.
  - For Romanian: Utilizes a pre-trained transformer model (`nlptown/bert-base-multilingual-uncased-sentiment`).
- **Dynamic Evaluation Metrics**: Updates metrics (Accuracy, Precision, Recall, F1 Score) as new user inputs are added.

## Requirements

Install the required Python libraries:

```bash
pip install nltk spacy langdetect transformers scikit-learn
python -m spacy download ro_core_news_sm
```

## Usage

1. **Run the Script**: Execute the Python file to start the sentiment analysis program.
2. **Input Text**: Enter text in English or Romanian to analyze its sentiment.
3. **View Results**:
   - The sentiment of the input text will be displayed.
   - Updated evaluation metrics (Accuracy, Precision, Recall, F1 Score) will be printed.

## Sample Workflow

1. Enter a phrase to analyze sentiment.
2. View the sentiment result (Positive/Negative/Neutral).
3. See updated evaluation metrics based on all previous inputs.
4. Repeat or exit the program.

