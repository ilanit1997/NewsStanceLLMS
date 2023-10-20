import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
import json
from tqdm import tqdm

class PerspectrumDataset(Dataset):
    def __init__(self, split='train'):
        """
        data available is:
        cId: claim id
        claim: claim text
        source: source of the claim
        topics: topics of the claim
        coarse_stance: coarse stance of the perspective (support, undermine)
        fine_grained_stance: fine grained stance of the perspective (support, mildly support, mildly undermine, undermine)
        pId: perspective id
        perspective: perspective text
        set: train, dev, test

        """
        claims_and_annotations = pd.read_json('data/perspectrum/perspectrum_with_answers_v1.0.json')
        claims_and_annotations = self.process_claims(claims_and_annotations)
        perspectives = pd.read_json('data/perspectrum/perspective_pool_v1.0.json')
        perspectives['perspective'] = perspectives['text']
        perspectives = perspectives[['pId', 'perspective']]
        data = claims_and_annotations.merge(perspectives, on='pId')
        with open('data/perspectrum/dataset_split_v1.0.json') as json_file:
            data_split = json.load(json_file)
        data = self.split_to_sets(data_split, data)
        self.data = data[data['set'] == split].reset_index(drop=True)

    def process_claims(self, df):
        records = []
        for index, row in df.iterrows():
            cId = row['cId']
            text = row['text']
            source = row['source']
            topics = row['topics']
            for item in row['perspectives']:
                coarse_stance = item['stance_label_3']
                fine_grained_stance = item['stance_label_5']
                if fine_grained_stance == 'NO_MAJORITY_LABEL':
                    continue
                for pid in item['pids']:
                    records.append(
                        {'cId': cId, 'claim': text, 'source': source, 'topics': topics, 'coarse_stance': coarse_stance,
                         'fine_grained_stance': fine_grained_stance, 'pId': pid})
        df = pd.DataFrame.from_records(records)
        return df

    def split_to_sets(self, data_split, data):
        data['set'] = data['cId'].apply(lambda cId: data_split[str(cId)])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_item = self.data.loc[item]
        return data_item['perspective'], data_item['claim'],data_item['fine_grained_stance']


class XStanceDataset(Dataset):
    """
    data available is:
    language: language of the comment (fr,it,de)
    question: question text
    comment: comment text
    topic: topic of the question and comment
    label: label of the stance of the comment towards the question, can be (AGAINST, FAVOR)
    """
    def __init__(self,split):
        self.data = load_dataset('x_stance',split=split)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]['language'],self.data[item]['question'],self.data[item]['comment'],self.data[item]['label']

class SemEval2016Dataset(Dataset):
    """
    data available is:
    id: id of the tweet
    Target: the topic of the tweet
        Atheism, Climate Change is a Real Concern, Feminist Movement, Hillary Clinton,
        Legalization of Abortion in train and dev
        Donald trump- in test
    Tweet: the tweet text
    Stance: the stance of the tweet towards the target
    (AGAINST, FAVOR, NONE)
    Sentiment: the sentiment of the tweet
    (POSITIVE, NEGATIVE, NEITHER)
    The labels are also written in the read me for further explanation
    """
    def __init__(self,split):
        if split == 'train':
            self.data = pd.read_csv('data/SemEval_2016/trainingdata-all-annotations.txt',delimiter='\t',encoding='latin-1')
        elif split == 'validation':
            self.data = pd.read_csv('data/SemEval_2016/testdata-taskA-all-annotations.txt',delimiter='\t',encoding='latin-1')
        elif split == 'test':
            self.data = pd.read_csv('data/SemEval_2016/testdata-taskB-all-annotations.txt',delimiter='\t',encoding='latin-1')
        else:
            raise ValueError('split must be train, dev or test')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        row = self.data.loc[0]
        return row['Tweet'],row['Target'],row['Stance']

