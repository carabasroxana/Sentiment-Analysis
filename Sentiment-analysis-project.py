import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from langdetect import detect
import spacy
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

analyzer_en = SentimentIntensityAnalyzer()

nlp_ro = spacy.load('ro_core_news_sm')

sentiment_analyzer_ro = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith('en'):
            return 'en'
        elif lang.startswith('ro'):
            return 'ro'
        else:
            return 'en'
    except:
        return 'en'

def preprocess_text(text, lang):
    tokens = word_tokenize(text.lower())

    if lang == 'en':
        lang_code = 'english'
    elif lang == 'ro':
        lang_code = 'romanian'
    else:
        lang_code = 'english'  # Default to English if language not recognized

    try:
        stop_words = stopwords.words(lang_code)
    except OSError:
        stop_words = []

    filtered_tokens = [token for token in tokens if token not in stop_words]

    if lang == 'en':
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    elif lang == 'ro':
        doc = nlp_ro(' '.join(filtered_tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
    else:
        lemmatized_tokens = filtered_tokens

    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def get_sentiment(text, lang):
    if lang == 'en':
        scores = analyzer_en.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    elif lang == 'ro':
        result = sentiment_analyzer_ro(text)[0]
        label = result['label']

        if '1' in label or '2' in label:
            sentiment = 'Negative'
        elif '4' in label or '5' in label:
            sentiment = 'Positive'
        else:
            sentiment = 'Neutral'
    else:
        sentiment = 'Neutral'
    return sentiment

sample_texts = [
    ("I love this product!", 'en', 'Positive'),
    ("This is the worst service ever.", 'en', 'Negative'),
    ("Este un serviciu excelent.", 'ro', 'Positive'),
    ("Nu mi-a plăcut deloc.", 'ro', 'Negative'),
    ("The product is okay, nothing special.", 'en', 'Neutral'),
    ("Produsul este acceptabil.", 'ro', 'Neutral')
]

actual_labels = []
predicted_labels = []

for text, lang, actual_sentiment in sample_texts:
    detected_lang = detect_language(text)
    processed_text = preprocess_text(text, detected_lang)
    predicted_sentiment = get_sentiment(processed_text, detected_lang)

    actual_labels.append(actual_sentiment)
    predicted_labels.append(predicted_sentiment)

def calculate_metrics(actual_labels, predicted_labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(actual_labels, predicted_labels)
        precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = calculate_metrics(actual_labels, predicted_labels)

print(f"Initial Evaluation Metrics:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\n")

while True:
    user_input = input("Enter a phrase to analyze sentiment: ")

    lang = detect_language(user_input)

    if lang not in ['en', 'ro']:
        print("Language not recognized")
        continue

    processed_text = preprocess_text(user_input, lang)
    sentiment = get_sentiment(processed_text, lang)
    print("The sentiment of the phrase is:", sentiment)

    actual_labels.append('Unknown')
    predicted_labels.append(sentiment)

    accuracy, precision, recall, f1 = calculate_metrics(actual_labels, predicted_labels)
    print(f"\nUpdated Evaluation Metrics (based on all inputs so far):\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")

    again = input("Do you want to analyze another phrase? (yes/no): ")
    if again.lower() != 'yes':
        print("Exiting the program.")
        break
