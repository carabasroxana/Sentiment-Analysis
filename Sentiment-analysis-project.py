import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (only need to do this once)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define the preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Define the get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment

# Loop to get input from the user
while True:
    user_input = input("Enter a phrase to analyze sentiment: ")
    # Preprocess the input text
    processed_text = preprocess_text(user_input)
    # Analyze the sentiment
    sentiment = get_sentiment(processed_text)
    # Output the sentiment
    print("The sentiment of the phrase is:", sentiment)
    print()  # Add an empty line for better readability

    # Ask if the user wants to analyze another phrase
    again = input("Do you want to analyze another phrase? (yes/no): ")
    if again.lower() != 'yes':
        print("Exiting the program.")
        break

# Preprocess the input text
processed_text = preprocess_text(user_input)

# Analyze the sentiment
sentiment = get_sentiment(processed_text)
