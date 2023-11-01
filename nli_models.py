import time

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from tqdm import tqdm
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import math
import torch
import json
import numpy as np
from itertools import permutations
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import roc_auc_score, f1_score
import re
import string
import gc

def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text


def iter_list(list_, batch_size):
    num_of_batches = math.ceil(len(list_) / batch_size)
    for i in range(num_of_batches):
        yield list_[i * batch_size:(i + 1) * batch_size]


class StanceDetectionNliModel():
    """
    based on Google paper "Stretching Sentence-pair NLI Models to Reason over Long Documents and Clusters"
    https://arxiv.org/pdf/2204.07447.pdf
    with adaptation to the case of neutral class included, as most nli models use neutral class
    """

    def __init__(self, model_name='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', batch_size=32, K=1,
                 device='cpu'):
        # self.positive_stances = ['The article is pro Israel', 'The article is pro Hamas',
        #                          'The article is pro Palestine']
        # self.negative_stances = ['The article is anti Israel', 'The article is anti Hamas',
        #                          'The article is anti Palestine']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.K = K
        self.device = device

    def split_doc(self, doc):
        # doc = remove_non_necessary_punctuation(doc)
        sentences = sent_tokenize(doc)
        sentences = [normalize_text(x) for x in sentences]
        return sentences

    def predict(self, hypotheses, premises):
        self.model.eval()
        with torch.no_grad():
            premises_hypothesis_pairs = [(f"premise:{premise}", f"hypothesis:{hypothesis}") for premise, hypothesis in
                                         zip(premises, hypotheses)]

            all_premises_probs = []
            for batch in iter_list(premises_hypothesis_pairs, self.batch_size):
                tokenized_input_seq_pair = self.tokenizer.batch_encode_plus(batch, return_tensors='pt',
                                                                            truncation='only_first',
                                                                            return_token_type_ids=True, max_length=512,
                                                                            padding=True)



                tokenized_input_seq_pair = tokenized_input_seq_pair
                input_ids = tokenized_input_seq_pair['input_ids'].long()
                # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
                token_type_ids = tokenized_input_seq_pair['token_type_ids'].long()
                attention_mask = tokenized_input_seq_pair['attention_mask'].long()
                outputs = self.model(input_ids.to(self.device),
                                     attention_mask=attention_mask.to(self.device),
                                     token_type_ids=token_type_ids.to(self.device),
                                     labels=None)

                predicted_probability = torch.softmax(outputs[0], dim=1)
                all_premises_probs += predicted_probability
        #gc.collect()
        return torch.stack(all_premises_probs).cpu().numpy()

    def retrieve_and_predict(self, docs, hypothesis):
        all_sentences = []
        limits = [0]
        for doc in docs:
            sentences = self.split_doc(doc)
            all_sentences += sentences
            limits.append(len(sentences) + limits[-1] if len(limits) > 0 else len(sentences))
        if len(all_sentences) == 0:
            return None
        probs = self.predict([hypothesis] * len(all_sentences), all_sentences)
        predictions = []
        for i in range(len(limits) - 1):
            doc_probs = probs[limits[i]:limits[i + 1]]
            doc_prediction = doc_probs.mean(axis=0)
            predictions.append(doc_prediction)
        return np.stack(predictions)
        # return {'max_prob_per_class': np.max(probs, axis=0), 'chosen_class': np.argmax(np.max(probs, axis=0))}

    def retrieve_and_rerank(self, docs, hypothesis):
        all_docs_sentences = []
        limits = [0]
        for doc in docs:
            sentences = self.split_doc(doc)
            all_docs_sentences += sentences
            limits.append(len(sentences) + limits[-1] if len(limits) > 0 else len(sentences))
        if len(all_docs_sentences) == 0:
            return None
        probs = self.predict([hypothesis] * len(all_docs_sentences), all_docs_sentences)
        # probs = self.predict([hypothesis]*len(sentences), sentences)
        # docs_chunks = []
        all_orders = list(permutations([0, 1, 2], 3))
        limits2 = [0]
        new_premises = []
        for i in range(len(limits) - 1):
            docs_probs = probs[limits[i]:limits[i + 1]]
            sentences = all_docs_sentences[limits[i]:limits[i + 1]]
            best_entitlement_sentences = [sentences[j] for j in np.argsort(docs_probs[:, 0])[::-1][:self.K]]
            best_neutral_sentences = [sentences[j] for j in np.argsort(docs_probs[:, 1])[::-1][:self.K]]
            best_contradiction_sentences = [sentences[j] for j in np.argsort(docs_probs[:, 2])[::-1][:self.K]]
            chosen_sentences = [best_entitlement_sentences, best_neutral_sentences, best_contradiction_sentences]
            doc_new_premises = []
            for order in all_orders:
                new_premise = ""
                for i in order:
                    new_premise += " ".join(chosen_sentences[i])
                doc_new_premises.append(new_premise)
            new_premises += doc_new_premises
            limits2.append(len(doc_new_premises) + limits2[-1] if len(limits2) > 0 else len(doc_new_premises))
        probs = self.predict([hypothesis] * len(new_premises), new_premises)
        docs_predictions = []
        for i in range(len(limits2) - 1):
            docs_probs = probs[limits2[i]:limits2[i + 1]]
            doc_prediction = docs_probs.mean(axis=0)
            docs_predictions.append(doc_prediction)
        return np.stack(docs_predictions)


def evaluate(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    datasetlabel_to_label = {'entailment': 0, 'not_entailment': 1}
    prediction_to_label_dict = {0: 0, 1: 1, 2: 1}
    all_predictions = []
    all_entailment_probs = []
    all_labels = []
    successful_docs = 0
    for batch in tqdm(dataloader):
        try:
            premise = batch['premise'][0]
            hypothesis = batch['hypothesis'][0]
            prediction = model.retrieve_and_rerank(premise, hypothesis)
            prediction_probs = prediction['prediction_probs']
            entailment_prob = prediction_probs[0]
            all_entailment_probs.append(entailment_prob)
            prediction_class = prediction['chosen_class']
            prediction = prediction_to_label_dict[prediction_class]
            all_predictions.append(prediction)
            label = batch['label'][0]
            all_labels.append(datasetlabel_to_label[label])
            successful_docs += 1
        except:
            continue
    print(f"Manged to evaluate {successful_docs} out of {len(dataset)} documents")
    acc = sum([1 if all_predictions[i] == all_labels[i] else 0 for i in range(len(all_predictions))]) / len(
        all_predictions)
    # For roc auc we need to supply it with binary labels and the probability to get the label 1
    # Therefore we need to convert all the probabilities to their complement
    roc_auc = roc_auc_score(all_labels, [1 - p for p in all_entailment_probs])
    f1 = f1_score(all_labels, all_predictions)

    return {'accuracy': acc, 'roc_auc': roc_auc, 'f1': f1}

#
# class ANLIdataset(Dataset):
#     def __init__(self):
#         anli_train_r1 = load_dataset('anli', split='train_r1')
#         anli_val_r1 = load_dataset('anli', split='dev_r1')
#         anli_test_r1 = load_dataset('anli', split='test_r1')
#         anli_train_r2 = load_dataset('anli', split='train_r2')
#         anli_val_r2 = load_dataset('anli', split='dev_r2')
#         anli_test_r2 = load_dataset('anli', split='test_r2')
#         anli_train_r3 = load_dataset('anli', split='train_r3')
#         anli_val_r3 = load_dataset('anli', split='dev_r3')
#         anli_test_r3 = load_dataset('anli', split='test_r3')
#         self.data = concatenate_datasets([anli_train_r1, anli_val_r1, anli_test_r1, anli_train_r2, anli_val_r2,
#                                           anli_test_r2, anli_train_r3, anli_val_r3, anli_test_r3])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         return self.data[item]
#
#
# anli_dataset = ANLIdataset()
# anli_premises = []
# anli_hypotheses = []
# for i in tqdm(range(len(anli_dataset))):
#     anli_premises.append(anli_dataset[i]['premise'])
#     anli_hypotheses.append(anli_dataset[i]['hypothesis'])
#
# model = StanceDetectionNliModel(device='cuda')
# train_dataset = load_dataset("saattrupdan/doc-nli", split='train')
# val_dataset = load_dataset("saattrupdan/doc-nli", split='val')
# test_dataset = load_dataset("saattrupdan/doc-nli", split='test')
# dataset = concatenate_datasets([train_dataset, val_dataset, test_dataset])
# num = 10000
# indexes = list(range(len(dataset)))
# np.random.shuffle(indexes)
# chosen_indexes = []
# for i in tqdm(indexes):
#     info = dataset[i]
#     premise = info['premise']
#     hypothesis = info['hypothesis']
#     if premise in anli_premises and hypothesis in anli_hypotheses:
#         continue
#     chosen_indexes.append(i)
#     if len(chosen_indexes) == num:
#         break
# # indexes = list(range(len(dataset)))
# # np.random.shuffle(indexes)
# subset_dataset = Subset(dataset,chosen_indexes)
# # x = pd.read_json('/data/home/yehonatan-pe/temp/DocNLI_dataset/train.json')
# # print(x.iloc[0])
# print(evaluate(model,subset_dataset))
