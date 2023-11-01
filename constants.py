from enum import Enum

class DataCols(Enum):
    NEWS_NAME = 'news name'
    HEADLINE = 'headline'
    DESCRIPTION = 'description'
    CONTENT = 'content'
    DATE_PUBLISHED = 'date_published'
    DATE_MODIFIED = 'date_modified'
    URL = 'url'
    LLAMA2_AFFILIATION = 'Llama2 Affiliation'
    NLTK_NEG = 'nltk_neg'
    NLTK_NEU = 'nltk_neu'
    NLTK_POS = 'nltk_pos'
    NLTK_COMPOUND = 'nltk_compound'
    NLTK_SENTIMENT_FROM_COMPOUND = 'NLTK Sentiment (from compound)'
    TOP_10_MOST_COMMON_WORDS = 'Top 10 most common words'
    FINANCIAL_SENTIMENT = 'Financial Sentiment'
    NEWS_SENTIMENT = 'News Sentiment'
    POLITICAL_AFFILIATION = 'Political Affiliation'

class ParsedDataCols(Enum):
    NLTK_BEST = "nltk_sent_max"
    NEWS_NAME = 'news name_parsed'
    HEADLINE = 'headline_parsed'
    DESCRIPTION = 'description_parsed'
    CONTENT = 'content_parsed'
    DATE_PUBLISHED = 'date_published_parsed'
    DATE_MODIFIED = 'date_modified_parsed'
    URL = 'url_parsed'
    LLAMA2_AFFILIATION = 'Llama2 Affiliation_parsed'
    NLTK_NEG = 'nltk_neg_parsed'
    NLTK_NEU = 'nltk_neu_parsed'
    NLTK_POS = 'nltk_pos_parsed'
    NLTK_COMPOUND = 'nltk_compound_parsed'
    NLTK_SENTIMENT_FROM_COMPOUND = 'NLTK Sentiment (from compound)_parsed'
    TOP_10_MOST_COMMON_WORDS = 'Top 10 most common words_parsed'
    FINANCIAL_SENTIMENT = 'Financial Sentiment_parsed'
    NEWS_SENTIMENT = 'News Sentiment_parsed'
    POLITICAL_AFFILIATION = 'Political Affiliation_parsed'


class ProClasses(Enum):
    PRO_HAMAS = "Pro-hamas"
    PRO_ISRAEL = "Pro-Israel"
    Neutral = "Neutral"
    OTHER = "Other"