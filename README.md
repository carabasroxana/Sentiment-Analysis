# Sentiment Analysis Command-Line Tool

This Python script allows you to input phrases and analyze their sentiment—whether they are **Positive**, **Negative**, or **Neutral**. It utilizes the Natural Language Toolkit (NLTK) library, specifically the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer, to perform the analysis.

## Features

- **Interactive User Input**: Continuously prompts the user to enter phrases for sentiment analysis.
- **Sentiment Classification**: Determines the sentiment based on the input text.
- **Text Preprocessing**: Cleans and preprocesses the input text by tokenizing, removing stop words, and lemmatizing.
- **Loop Control**: Offers the option to analyze multiple phrases in one session.

## Prerequisites

- **Python 3.x**
- **NLTK library**

## Installation

1. **Clone the repository or download the script** to your local machine.

2. **Install the NLTK library** (if not already installed):

   ```bash
   pip install nltk
   ```

3. **Download Necessary NLTK Data Packages**:

   The script includes commands to download the required NLTK data packages. When you run the script for the first time, it will download:

   - `vader_lexicon`
   - `punkt`
   - `stopwords`
   - `wordnet`

   If you've already downloaded these packages, you can comment out or remove the `nltk.download()` lines in the script.

## Usage

1. **Run the Script**:

   Navigate to the directory containing the script and execute:

   ```bash
   python sentiment_analysis.py
   ```

2. **Enter a Phrase**:

   When prompted, input the phrase you wish to analyze.

3. **View the Sentiment Result**:

   The script will display whether the sentiment of the phrase is Positive, Negative, or Neutral.

4. **Analyze Additional Phrases**:

   After each analysis, the script will ask if you want to analyze another phrase. Type `yes` to continue or `no` to exit.

## Example Session

```
Enter a phrase to analyze sentiment: I absolutely love this new app!
The sentiment of the phrase is: Positive

Do you want to analyze another phrase? (yes/no): yes
Enter a phrase to analyze sentiment: I am not happy with the service.
The sentiment of the phrase is: Negative

Do you want to analyze another phrase? (yes/no): yes
Enter a phrase to analyze sentiment: It's an average experience.
The sentiment of the phrase is: Neutral

Do you want to analyze another phrase? (yes/no): no
Exiting the program.
```

## How It Works

### Text Preprocessing

The script preprocesses the input text to improve the accuracy of the sentiment analysis:

1. **Tokenization**: Splits the text into individual words (tokens).
2. **Lowercasing**: Converts all text to lowercase to ensure consistency.
3. **Stop Words Removal**: Eliminates common words that do not contribute significant meaning (e.g., "and", "the", "is").
4. **Lemmatization**: Reduces words to their base or root form (e.g., "running" becomes "run").

### Sentiment Analysis

- **VADER Sentiment Analyzer**: Uses NLTK's `SentimentIntensityAnalyzer` to compute sentiment scores.
- **Compound Score**: The script evaluates the `compound` score, which ranges from -1 (most negative) to +1 (most positive).
  - **Positive**: `compound >= 0.05`
  - **Negative**: `compound <= -0.05`
  - **Neutral**: `-0.05 < compound < 0.05`


