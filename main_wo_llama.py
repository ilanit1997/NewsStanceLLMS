import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset

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
all_datasets = [NewsDataset(file_path) for file_path in data_file_paths]
final_dataset = ConcatDataset(all_datasets)

def custom_collate_fn(batch):
    text_data, metadata = zip(*batch)
    return text_data, metadata  # Returns a tuple of texts and a tuple of metadata dicts

data_loader = DataLoader(final_dataset, batch_size=10, shuffle=False, collate_fn=custom_collate_fn)
all_results = []

for batch_text, batch_metadata in data_loader:
    for data, metadata  in zip(batch_text, batch_metadata):
        full_text = f"{data.get('headline')} {data.get('description')} {data.get('content')}"
        result = metadata

        model_results = {
            "NLTK Sentiment Scores": sentiment_model.analyze(full_text),
            "NLTK Sentiment (from compound)": sentiment_model.analyze_compound(full_text),
            "Top 10 most common words": text_model.frequency_analysis(full_text),
            "Financial Sentiment": finance_sentiment_model.analyze(full_text),
            "News Sentiment": multi_news_sentiment_model.analyze(full_text),
            "Political Affiliation": political_model.analyze(full_text),
        }
        result.update(model_results)

        all_results.append(result)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(all_results)

# Save DataFrame to CSV
csv_path = "output/results.csv"
df.to_csv(csv_path, index=False)