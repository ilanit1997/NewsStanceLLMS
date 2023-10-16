## README.md

# News Article Analyzer

## Overview

This module provides functionalities to analyze news articles in various capacities such as sentiment analysis, frequency analysis, and classification based on specific criteria.

### Main Components:

1. **DataReader**: Loads and processes the news article data.
2. **SentimentAnalyzer**: Analyzes the sentiment of a given text.
3. **TextAnalyzer**: Cleans and processes the text for word frequency analysis.
4. **ClassifierModel**: Classifies the text based on a pre-trained model.
5. **ArticleScorer**: Evaluates articles based on their alignment with specific word groups.
6. **LlamaChatModel**: A generative model to produce chat-like responses based on article data.

## Requirements:

- nltk
- transformers
- torch

## Setting Up:

1. Ensure you have the required Python libraries installed:
    ```bash
    pip install nltk transformers torch
    ```

2. Before using the Sentiment Analyzer and Text Analyzer functionalities, ensure to download the required NLTK resources:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

3. For the `ClassifierModel` to function, you need to have a configuration file named `models_config.json` in the `configs` directory. This configuration file should contain model details.

4. `LlamaChatModel` requires a configuration file and prompt file as specified in `models_config.json`.

## Usage:

**Loading Data:**
```python
from models import DataReader

data = DataReader.load_data('path_to_data.json')
processed_data = DataReader.process_data(data)
```

**Sentiment Analysis:**
```python
from models import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze(processed_data)
```

**Text Analysis:**
```python
from models import TextAnalyzer

analyzer = TextAnalyzer()
cleaned_text = analyzer.clean_text(processed_data)
word_freq = analyzer.frequency_analysis(cleaned_text)
```

**Article Classification:**
```python
from models import ClassifierModel, ModelType

classifier = ClassifierModel(ModelType.FinancialSentimentModel)
category = classifier.analyze(processed_data)
```

**Article Scoring:**
```python
from models import ArticleScorer

scorer = ArticleScorer()
score = scorer.score_article(processed_data)
```

**Generate Chat Response:**
```python
from models import LlamaChatModel

llama_chat = LlamaChatModel()
response = llama_chat.generate_text(data)
```

## Important Notes:

- Ensure the `configs` directory has the necessary configuration files for the models.
- When using `LlamaChatModel`, ensure the availability of the associated pre-trained model either online or locally.



---