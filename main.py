from models import *

data = DataReader.load_data('data/bbc_test.json')
full_text = DataReader.process_data(data)

results = {}
results['NLTK Sentiment Scores'] = SentimentAnalyzer().analyze(full_text)
results['NLTK Sentiment (from compound)'] = SentimentAnalyzer().analyze_compound(full_text)
results['Top 10 most common words'] = TextAnalyzer().frequency_analysis(full_text)
finance_sentiment_model = ClassifierModel(ModelType.FinancialSentimentModel)
political_model = ClassifierModel(ModelType.PoliticalAffiliationModel)
results['Financial Sentiment'] = finance_sentiment_model.analyze(full_text)
multi_news_sentiment_model = ClassifierModel(ModelType.MultiNewsSentimentModel)
results['News Sentiment'] = multi_news_sentiment_model.analyze(full_text)
results['Political Affiliation'] = political_model.analyze(full_text)

## doesnt work as expected - return similar scores for both pro-israel and pro-hamas
# article_score = ArticleScorer()
# results['Group Analysis'] = article_score.score_article(full_text)

##llama2
llama2_model = LlamaChatModel(config_path="configs/models_config.json")
results['Political Affiliation'] = llama2_model.generate_text(data)

# Print consolidated results
for key, value in results.items():
    print(f"{key}: {value}\n")