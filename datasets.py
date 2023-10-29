import json
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, file_path):
        self.data_file = self.load_data_file(file_path)
        self.news_name = file_path.split("/")[-1].split(".")[0]

    @staticmethod
    def load_data_file(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def process_data_item(data: dict) -> str:
        full_text = f"{data.get('headline')} {data.get('description')} {data.get('content')}"
        return full_text

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        data = self.data_file[idx]
        data_text = {"headline": data.get('headline'),
                     "description": data.get('description'),
                     "content": data.get('content')}
        metadata = self.get_metadata(data)
        return data_text, metadata

    def get_metadata(self, data: dict) -> dict:
        # Extract the required data fields
        headline = data.get("headline")
        url = data.get("url")
        description = data.get("description")
        content = data.get("content")
        timestamp = data.get("timestamp")
        # authors = [""]
        # if type(data.get("author")) == list:
        #     authors = [x.get("name") for x in data.get("author")]
        # elif type(data.get("author")) == dict:
        #     authors = [data.get("author").get("name")]
        # authors = ",".join(authors)

        # Get results
        results = {
            "news name": self.news_name,
            "headline": headline,
            "description": description,
            "content": content,
            "timestamp": timestamp,
            # "author": authors,
            "url": url,
        }

        return results