import pandas as pd
import string
import ast
import numpy as np


def remove_punctuation(input_string):
    # Create a translation table to remove all punctuation characters
    translator = str.maketrans('', '', string.punctuation)

    # Use translate() to remove punctuation from the input string
    cleaned_string = input_string.translate(translator)

    return cleaned_string


def llama_preprocess_by_first_entry(df: pd.DataFrame):
    llama2 = [str(x).lower() for x in df['llama2'].tolist()]
    extracted_llama_2 = [remove_punctuation(x.split(',')[0]).strip() for x in llama2]
    extracted_llama_2_processed = []
    for x in extracted_llama_2:
        if 'natural' in x:
            extracted_llama_2_processed.append('natural')
        elif "prohamas" in x or 'propalestine' in x or "propalestinian" in x:
            extracted_llama_2_processed.append("pro-hamas")
        elif "proisrael" in x:
            extracted_llama_2_processed.append("pro-israel")
        else:
            extracted_llama_2_processed.append(None)
    df['llama2_processed'] = extracted_llama_2_processed
    return df


def llama_preprocess_only_full_agreement(df: pd.DataFrame):
    llama2 = [remove_punctuation(str(x).lower()) for x in df['llama2'].tolist()]
    extracted_llama_2 = []
    for x in llama2:
        list_of_labels = []
        if 'natural' in x:
            list_of_labels.append('natural')
        if "proisrael" in x:
            list_of_labels.append("pro-israel")
        if "prohamas" in x or 'propalestine' in x or "propalestinian" in x:
            list_of_labels.append("pro-hamas")
        extracted_llama_2.append(list_of_labels)
    for i in range(len(extracted_llama_2)):
        if len(extracted_llama_2[i]) == 0:
            extracted_llama_2[i] = None
        elif len(extracted_llama_2[i]) > 1:
            extracted_llama_2[i] = None
        else:
            extracted_llama_2[i] = extracted_llama_2[i][0]
    df['llama2_ant'] = extracted_llama_2
    return df

def check_if_all_duplicates_are_the_same(seperated_annotations):
    for x in seperated_annotations:
        are_the_same = []
        for i in range(len(x) - 1):
            are_the_same.append(x[i] == x[i + 1])
        if False in are_the_same:
            return False
    return True


def process_nltk_sentiment_scores(df):
    scores = df['NLTK Sentiment Scores'].tolist()
    seperated_annotations = [x.split('", "') for x in scores]
    for x in seperated_annotations:
        for i in range(len(x)):
            x[i] = x[i].replace('"', '')
            x[i] = x[i].replace('[', '')
            x[i] = x[i].replace(']', '')
    if not check_if_all_duplicates_are_the_same(seperated_annotations):
        raise ValueError("Not all annotators are the same")
    seperated_annotations = [ast.literal_eval(x[0]) for x in seperated_annotations]
    annotations = []
    for x in seperated_annotations:
        labels_ids = ['negative', 'neutral', 'positive']
        label = labels_ids[np.argmax([x['neg'], x['neu'], x['pos']])]
        annotations.append(label)
    df['nltk_sent_ant'] = annotations
    return df

def process_nltk_sentiment_scores_from_compound(df):
    scores = df['NLTK Sentiment (from compound)'].tolist()
    seperated_annotations = [x.split(',') for x in scores]
    for x in seperated_annotations:
        for i in range(len(x)):
            x[i] = remove_punctuation(x[i]).strip()
    if not check_if_all_duplicates_are_the_same(seperated_annotations):
        raise ValueError("Not all annotators are the same")
    seperated_annotations = [x[0].lower() for x in seperated_annotations]
    df['nltk_sent_ant_comp'] = seperated_annotations
    return df


def process_financial_sentiment_scores(df):
    scores = df['Financial Sentiment']
    seperated_annotations = [x.split(',') for x in scores]
    for x in seperated_annotations:
        for i in range(len(x)):
            x[i] = remove_punctuation(x[i]).strip()
    if not check_if_all_duplicates_are_the_same(seperated_annotations):
        raise ValueError("Not all annotators are the same")
    seperated_annotations = [x[0].lower() for x in seperated_annotations]
    df['financial_sent_ant'] = seperated_annotations
    return df


def process_news_sentiment_scores(df):
    scores = df['News Sentiment'].tolist()
    seperated_annotations = [x.split(',') for x in scores]
    for x in seperated_annotations:
        for i in range(len(x)):
            x[i] = remove_punctuation(x[i]).strip()
    if not check_if_all_duplicates_are_the_same(seperated_annotations):
        raise ValueError("Not all annotators are the same")
    seperated_annotations = [x[0].lower() for x in seperated_annotations]
    df['news_sent_ant'] = seperated_annotations
    return df


def process_political_affiliation_scores(df):
    scores = df['Political Affiliation'].tolist()
    seperated_annotations = [x.split(',') for x in scores]
    for x in seperated_annotations:
        for i in range(len(x)):
            x[i] = remove_punctuation(x[i]).strip()
    if not check_if_all_duplicates_are_the_same(seperated_annotations):
        raise ValueError("Not all annotators are the same")
    seperated_annotations = [x[0].lower() for x in seperated_annotations]
    df['political_aff_ant'] = seperated_annotations
    return df


def preprocess(df):
    df = llama_preprocess_only_full_agreement(df)
    df = process_nltk_sentiment_scores(df)
    df = process_nltk_sentiment_scores_from_compound(df)
    df = process_financial_sentiment_scores(df)
    df = process_news_sentiment_scores(df)
    df = process_political_affiliation_scores(df)
    return df
