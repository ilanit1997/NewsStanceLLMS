import os
import pandas as pd
import torch.cuda
from tqdm import tqdm
import gc

from models import *
from torch.utils.data import DataLoader, ConcatDataset
from CustomDatasets import NewsDataset

print(f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu')

data_folder = "data/20231028"

llama2_model = LlamaChatModel()
data_file_paths = [os.path.join(data_folder, news) for news in os.listdir(data_folder)]
all_datasets = [NewsDataset(file_path) for file_path in data_file_paths]
final_dataset = ConcatDataset(all_datasets)

def custom_collate_fn(batch):
    text_data, metadata = zip(*batch)
    return text_data, metadata  # Returns a tuple of texts and a tuple of metadata dicts

data_loader = DataLoader(final_dataset, batch_size=5, shuffle=False, collate_fn=custom_collate_fn)
all_results = []

for batch_text, batch_metadata in tqdm(data_loader, desc="Processing batches"):
    for i, model_result in enumerate(llama2_model.generate_text_batch(batch_text)):
        result = batch_metadata[i]
        result.update({"Llama2 Affiliation": model_result})
        all_results.append(result)

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    df = pd.DataFrame(all_results)

    # Save DataFrame to CSV
    csv_path = f"/data/home/ilanit.sobol/politics/output/results_llama2-7b.csv"
    df.to_csv(csv_path, index=False)