import json

import transformers
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, \
    RobertaTokenizer, RobertaModel
import torch
from enum import Enum
from torch.nn.functional import cosine_similarity

from preprocessing import *

# # Ensure you've downloaded the required resources
nltk.download('stopwords')
nltk.download('wordnet')

from torch import cuda, bfloat16

class DataReader:
    @staticmethod
    def load_data(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def process_data(data: dict) -> str:
        full_text = f"{data['description']} {data['headline']} {data['content']}"
        return full_text

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        polarities = self.sia.polarity_scores(text)
        return polarities

    def analyze_compound(self, text, threhold=0.05):
        polarities = self.sia.polarity_scores(text)
        compound_value = polarities["compound"]
        if compound_value > threhold:
            return "Positive"
        if compound_value < -threhold:
            return "Negative"
        else:
            return "Neutral"


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
class ClassifierModel:
    def __init__(self, model_type: ModelType):
        # Load the config
        with open('models_config.json', 'r') as file:
            config_file = json.load(file)
        if model_type.value not in config_file.keys():
            raise NotImplementedError()
        config = config_file[model_type.value]
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


class ArticleScorer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()

        # Precompute mean embeddings for Pro-Israel and Pro-Palestine words
        with open('israel_palestine_words.json', 'r') as file:
            stance_data = json.load(file)

        self.pro_israel_mean_embedding = self.get_mean_embedding(stance_data["Pro-Israel words"])
        self.pro_palestine_mean_embedding = self.get_mean_embedding(stance_data["Pro-Palestine words"])

        self.synonyms_dict = {"Israel": stance_data["Israel"], "Hamas": stance_data["Hamas"] }

    def get_mean_embedding(self, words):
        embeddings = [self.get_embedding(word) for word in words]
        return torch.stack(embeddings).mean(dim=0)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs['last_hidden_state'].mean(dim=1)

    @staticmethod
    def is_subsequence_of_tokens(synonym_tokens, token_list):
        """
        Check if synonym_tokens are a consecutive subsequence of token_list.
        """
        cleaned_token_list = [t.replace('Ġ', '') for t in token_list]

        for i in range(len(cleaned_token_list) - len(synonym_tokens) + 1):
            full_token = cleaned_token_list[i:i + len(synonym_tokens)]
            if full_token == synonym_tokens:
                return True
        return False
    @staticmethod
    def find_subtoken_indices(subtokens, tokens):
        """Return a list of starting indices of subtoken sequences in tokens."""
        indices = []
        cleaned_token_list = [t.replace('Ġ', '') for t in tokens]
        cleaned_subtokens_list = [t.replace('Ġ', '') for t in subtokens]

        for i in range(len(cleaned_token_list) - len(cleaned_subtokens_list) + 1):
            current_tokens = cleaned_token_list[i:i + len(cleaned_subtokens_list)]
            if current_tokens == cleaned_subtokens_list:
                indices.append(i)  # Just append the starting index
        return indices
    def extract_contextual_embeddings(self, article, synonym):
        # Preprocess the article and the synonym
        preprocessed_article = lemmatize_text(stem_text(article.lower()))
        preprocessed_synonym = lemmatize_text(stem_text(synonym.lower()))

        # Split the preprocessed article into sentences
        preprocessed_sentences = nltk.sent_tokenize(preprocessed_article)
        sentences = nltk.sent_tokenize(article)

        # Tokenize the synonym and the sentences
        synonym_tokens = self.tokenizer.tokenize(preprocessed_synonym)
        sentence_tokens = [self.tokenizer.tokenize(s) for s in preprocessed_sentences]

        # Find the sequences of tokens in the sentences that match the synonym tokens
        occurrences = []
        for i, tokens in enumerate(sentence_tokens):
            result_bool = self.is_subsequence_of_tokens(synonym_tokens, tokens)
            if result_bool:
                occurrences.append(i)

        embeddings = []

        for occ in occurrences:
            # Get one sentence before, the current sentence, and one sentence after
            context_sentences = sentences[max(0, occ - 1): min(len(sentences), occ + 2)]
            context = " ".join(context_sentences)

            # Convert the context to input features
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, padding=True)

            # Get the embedding of the synonym within this context
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs['last_hidden_state']

            # Locate the synonym within the context and extract its embedding
            context_tokens = self.tokenizer.tokenize(context)
            curr_synonym_tokens = self.tokenizer.tokenize(synonym)
            matched_indices = self.find_subtoken_indices(curr_synonym_tokens, context_tokens)

            if matched_indices:
                # Here, instead of just taking one position, take the mean of the embeddings of all the positions
                # corresponding to the subtokens of the word. This is assuming that you want an average representation.
                synonym_embedding = last_hidden_states[0, matched_indices].mean(dim=0)
                embeddings.append(synonym_embedding)

        return torch.stack(embeddings).mean(dim=0) if embeddings else None


    def compute_similarity(self, embedding):
        pro_israel_similarity = cosine_similarity(embedding, self.pro_israel_mean_embedding)
        pro_palestine_similarity = cosine_similarity(embedding, self.pro_palestine_mean_embedding)
        return pro_israel_similarity, pro_palestine_similarity

    def score_article(self, article):
        scores = []

        for group, synonyms in self.synonyms_dict.items():
            for synonym in synonyms:
                contextual_embedding = self.extract_contextual_embeddings(article, synonym)
                if contextual_embedding is not None:
                    contextual_embedding = contextual_embedding.view(1, -1)
                    pro_israel_similarity, pro_palestine_similarity = self.compute_similarity(contextual_embedding)
                    scores.append({"Group": group, "Synonym": synonym,
                                   "Pro-Israel Score": pro_israel_similarity.item(),
                                   "Pro-Palestine Score": pro_palestine_similarity.item()})

        return scores


class LlamaChatModel:
    def __init__(self, model_id='meta-llama/Llama-2-13b-chat-hf', prompt_file_path="prompt_llama2.txt"):
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        # Quantization configuration
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map='auto'
        )
        self.model.eval()

        self.generator = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=10,
            repetition_penalty=1.1
        )

        # Load prompts and data
        with open(prompt_file_path, "r") as file:
            self.prompt = "\n".join(file.readlines())

    def generate_text(self, data):
        prompt_full = self.prompt.replace("headline", data["headline"])
        prompt_full = prompt_full.replace("article", data["context"])

        results = self.generator(prompt_full, max_length=10, temperature=0.1, num_return_sequences=1)

        generated_texts = [result["generated_text"] for result in results]
        return generated_texts[0]