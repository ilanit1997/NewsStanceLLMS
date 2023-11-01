import json
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, file_path, max_chars=6000, prompt= "", filter_duplicates=True):
        self.data_file = self.load_data_file(file_path)
        if filter_duplicates:
            self.data_file = self.filter_duplicates()
        self.news_name = file_path.split("/")[-1].split(".")[0]
        self.max_chars = max_chars
        self.prompt = prompt

    @staticmethod
    def load_data_file(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def filter_duplicates(self):
        seen_entries = set()
        unique_data = []  # List to store non-duplicate entries

        for entry in self.data_file:
            unique_identifier = (entry['description'], entry['headline'], entry['main_url'], entry['content'])

            # If this combination hasn't been seen before, add it to the unique_data list
            if unique_identifier not in seen_entries:
                seen_entries.add(unique_identifier)
                unique_data.append(entry)

        # Replace the original data_file with the filtered unique_data
        return unique_data


    @staticmethod
    def process_data_item(data: dict) -> str:
        full_text = f"{data.get('headline')} {data.get('description')} {data.get('content')}"
        return full_text

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        data = self.data_file[idx]
        metadata = self.get_metadata(data)
        if self.prompt:
            try:
                content = data.get("content", "No content")
                truncated_content = content[:self.max_chars]
                headline = data.get("headline", "No headline")
                prompt = self.prompt.replace("{headline}", headline)
                prompt = prompt.replace("{article_text}", truncated_content)
            except Exception as e:
                print(f"Error processing data at index {idx}: {str(e)}")  # log or print error for debugging
                prompt = ""
            return prompt, metadata

        else:
            full_text = f"{data.get('headline', 'No headline')} \n {data.get('description', 'No description')} \n {data.get('content', 'No content')}"
            return full_text, metadata

    def get_metadata(self, data: dict) -> dict:
        headline = data.get("headline")
        url = data.get("url")
        description = data.get("description")
        content = data.get("content")
        date_published = data.get("date_published")
        date_modified = data.get("date_modified")
        results = {
            "news name": self.news_name,
            "headline": headline,
            "description": description,
            "content": content,
            "date_published": date_published,
            "date_modified": date_modified,
            "url": url,
        }

        return results