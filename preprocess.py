import pandas as pd
import string
import ast
import numpy as np
from constants import DataCols, ParsedDataCols, ProClasses


def remove_punctuation(input_string):
    # Create a translation table to remove all punctuation characters
    translator = str.maketrans('', '', string.punctuation)

    # Use translate() to remove punctuation from the input string
    cleaned_string = input_string.translate(translator)

    return cleaned_string



def llama_preprocess_only_full_agreement(df: pd.DataFrame):
    llama2 = [remove_punctuation(str(x).lower()) for x in df[DataCols.LLAMA2_AFFILIATION.value].tolist()]
    pro_hamas_words  = ["prohamas", 'propalestine', "propalestinian", "prohezbollah"]
    pro_israel_words  = ["proisrael"]
    neutral_words  = ["neutral"]

    extracted_llama_2 = []
    for x in llama2:
        list_of_labels = []
        if any(w in x for w in neutral_words):
            list_of_labels.append(ProClasses.Neutral.value)
        elif any(w in x for w in pro_israel_words):
            list_of_labels.append(ProClasses.PRO_ISRAEL.value)
        elif any(w in x for w in pro_hamas_words):
            list_of_labels.append(ProClasses.PRO_HAMAS.value)
        extracted_llama_2.append(list_of_labels)
    for i in range(len(extracted_llama_2)):
        if len(extracted_llama_2[i]) == 0:
            extracted_llama_2[i] = ProClasses.OTHER.value
        else:
            extracted_llama_2[i] = extracted_llama_2[i][0]
    df[ParsedDataCols.LLAMA2_AFFILIATION.value] = extracted_llama_2
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
    annotations = []
    for i, row in df.iterrows():
        labels_ids = ['Negative', 'Neutral', 'Positive']
        label = labels_ids[np.argmax([row[DataCols.NLTK_NEG.value], row[DataCols.NLTK_NEU.value], row[DataCols.NLTK_POS.value]])]
        annotations.append(label)
    df[ParsedDataCols.NLTK_BEST.value] = annotations
    return df


def preprocess_df(df):
    df = llama_preprocess_only_full_agreement(df)
    df = process_nltk_sentiment_scores(df)
    return df