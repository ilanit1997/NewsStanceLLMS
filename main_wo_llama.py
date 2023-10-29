import os
import pandas as pd
from tqdm import tqdm

from models import *

print(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')

data_folder = "data/20231028"

# llama2_model = LlamaChatModel()
sentiment_model = SentimentAnalyzer()
text_model = TextAnalyzer()
finance_sentiment_model = ClassifierModel(ModelType.FinancialSentimentModel)
political_model = ClassifierModel(ModelType.PoliticalAffiliationModel)
multi_news_sentiment_model = ClassifierModel(ModelType.MultiNewsSentimentModel)

data_file_paths = [os.path.join(data_folder, news) for news in os.listdir(data_folder)]
all_results = []

for file_path in data_file_paths:
    Reader = DataReader(file_path)
    print(f"working on: {Reader.news_name}")
    for data in tqdm(Reader.get_next_data(), total=len(Reader)):
        full_text = DataReader.process_data_item(data)

        # Extract the required data fields
        headline = data.get("headline")
        url = data.get("url")
        description = data.get("description")
        content = data.get("content")
        timestamp =  data.get("timestamp")
        authors = [""]
        if type(data.get("author")) == list:
            authors = [x.get("name") for x in data.get("author")]
        elif type(data.get("author")) == dict:
            authors = [data.get("author").get("name")]
        authors = ",".join(authors)

        # Get model results
        results = {
            "news": Reader.news_name,
            "headline": headline,
            "description": description,
            "content": content,
            "timestamp": timestamp,
            "author": authors,
            "url": url,

            "NLTK Sentiment Scores": sentiment_model.analyze(full_text),
            "NLTK Sentiment (from compound)": sentiment_model.analyze_compound(full_text),
            "Top 10 most common words": text_model.frequency_analysis(full_text),
            "Financial Sentiment": finance_sentiment_model.analyze(full_text),
            "News Sentiment": multi_news_sentiment_model.analyze(full_text),
            "Political Affiliation": political_model.analyze(full_text),
            # "Llama2 Affiliation": llama2_model.generate_text(data) ## to heavy
        }

        all_results.append(results)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(all_results)

# Save DataFrame to CSV
csv_path = "output/results.csv"
df.to_csv(csv_path, index=False)