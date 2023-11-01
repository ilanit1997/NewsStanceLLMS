import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from NewsDatasets import *
from models import *
import gc

print(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')

latest_date = "20231031"
data_folder = f"data/{latest_date}"

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
    return text_data, metadata

data_loader = DataLoader(final_dataset, batch_size=50, shuffle=False, collate_fn=custom_collate_fn)
all_results = []

for batch_text, batch_metadata in tqdm(data_loader, desc="Processing batches"):
    for full_text, metadata in zip(batch_text, batch_metadata):
        result = metadata
        sentiment_scores = sentiment_model.analyze(full_text)
        result.update(sentiment_scores)
        model_results = {
            "NLTK Sentiment (from compound)": sentiment_model.analyze_compound(full_text),
            "Top 10 most common words": text_model.frequency_analysis(full_text),
            "Financial Sentiment": finance_sentiment_model.analyze(full_text),
            "News Sentiment": multi_news_sentiment_model.analyze(full_text),
            "Political Affiliation": political_model.analyze(full_text),
        }
        result.update(model_results)

        all_results.append(result)

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(all_results)

# Save DataFrame to CSV
output_folder = f"output/{latest_date}"
os.makedirs(output_folder, exist_ok=True)
csv_path = f"{output_folder}/results_baselines.csv"
df.to_csv(csv_path, index=False)