import json
import re
from typing import List
import nltk
import transformers
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from enum import Enum
import torch


from torch import cuda, bfloat16


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    def analyze(self, text):
        polarities = self.sia.polarity_scores(text)
        polarities_copy = {}
        for key, value in polarities.items():
            polarities_copy[f"nltk_{key}"] = value
        return polarities_copy

    def analyze_compound(self, text, threhold=0.05):
        polarities = self.sia.polarity_scores(text)
        compound_value = polarities["compound"]
        if compound_value > threhold:
            return "Positive"
        if compound_value < -threhold:
            return "Negative"
        else:
            return "Neutral"
    def __str__(self):
        return ModelType.FinancialSentimentModel.value

class TextAnalyzer:
    def __init__(self):
        # Initialize stopwords and lemmatizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuations
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove stopwords and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        # Join back into a string
        text = ' '.join(tokens)
        # Remove extra whitespaces
        text = text.strip()
        return text

    def frequency_analysis(self, text: str, n_top=10):
        clean_text = self.clean_text(text)
        tokens = nltk.word_tokenize(clean_text)
        fdist = FreqDist(tokens)
        most_common_words_tuples = fdist.most_common(n_top)
        most_common_words = [x[0] for x in most_common_words_tuples]
        most_common_words_str = ", ".join(most_common_words)
        return most_common_words_str


class ModelType(Enum):
    FinancialSentimentModel = "FinancialSentimentModel"
    PoliticalAffiliationModel = "PoliticalAffiliationModel"
    MultiNewsSentimentModel = "MultiNewsSentimentModel"
    LlamaModel = "LlamaModel"

class ClassifierModel:
    def __init__(self, model_type: ModelType,
                 config_path: str = 'configs/models_config.json'):
        # Load the config
        with open(config_path, 'r') as file:
            config_file = json.load(file)
        if model_type.value not in config_file.keys():
            raise NotImplementedError()
        config = config_file[model_type.value]
        self.model_type = model_type.value
        self.model_name = config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

        if config["base_model"] == "bert":
            base_model_type = BertForSequenceClassification
        elif config["base_model"] == "roberta":
            base_model_type = RobertaForSequenceClassification
        else:
            raise NotImplementedError(f"{config['base_model']} hasn't been supported!")

        # Try to load the model from PyTorch weights. If that fails, attempt to load from TensorFlow weights.
        try:
            self.model = base_model_type.from_pretrained(self.model_name)
        except OSError:
            print(f"Trying to load {self.model_name} from TensorFlow weights.")
            self.model = base_model_type.from_pretrained(self.model_name, from_tf=True)
        self.max_length = 512
        self.categories = config["categories"]

    def _chunk_sentences(self, text):
        # Split the text into sentences
        sentences = nltk.sent_tokenize(text)
        chunks = []
        chunk = []
        chunk_len = 0

        for sentence in sentences:
            sentence_len = len(self.tokenizer.tokenize(sentence))
            if chunk_len + sentence_len <= self.max_length:
                chunk.append(sentence)
                chunk_len += sentence_len
            else:
                chunks.append(' '.join(chunk))
                chunk = [sentence]
                chunk_len = sentence_len
        if chunk:
            chunks.append(' '.join(chunk))
        return chunks

    def analyze(self, text):
        chunks = self._chunk_sentences(text)

        logits_list = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, padding='max_length',
                                        max_length=self.max_length)
                outputs = self.model(**inputs)
                logits = outputs.logits  # Assuming the logits are the first item in the outputs tuple
                logits_list.append(logits)

        # Average the logits
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        return self.categories[torch.argmax(avg_logits)]

    def __str__(self):
        return self.model_type


class LlamaChatModel:
    def __init__(self, config_path: str = "configs/models_config.json"):
        with open(config_path, 'r') as file:
            config_file = json.load(file)
            config = config_file.get(ModelType.LlamaModel.value)

        self.model_id = config["model_id"]
        prompt_file_path = config["prompt_file_path"]

        # Set quantization configuration
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model with quantization
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map='auto',
        )

        self.pipeline_type = config["pipeline_type"]

        self.pipeline = transformers.pipeline(
            self.pipeline_type,
            model=self.model,
            tokenizer=self.tokenizer,  # Explicitly provide the tokenizer here
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Load prompts
        with open(prompt_file_path, "r") as file:
            self.prompt = "\n".join(file.readlines())

    def generate_text(self, data: dict) -> List:
        prompt_full = self.prompt.replace("{headline}", data["headline"])
        prompt_full = prompt_full.replace("{article_text}", data["content"])

        num_tokens = len(self.tokenizer.tokenize(prompt_full))
        results = self.pipeline(
            prompt_full,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=3500,
        )
        prompt_len = len(prompt_full)
        output = [result["generated_text"][prompt_len:].strip() for result in results]
        return output

    def generate_text_batch(self, data_batch):

        outputs = self.pipeline(
            list(data_batch),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return outputs


    def __str__(self):
        return ModelType.LlamaModel.value

    def post_process(self, outputs):
        decoded_outputs = []
        for output_batch in outputs:
            output = output_batch[0].get("generated_text")
            instruction_end = output.find("[/INST]")

            if instruction_end != -1:  # If the pattern is found
                instruction_end += len("[/INST]")
                current_output = output[instruction_end:]
                decoded_outputs.append(current_output)
            else:
                decoded_outputs.append(output)

        return decoded_outputs