import numpy as np
import pandas as pd
import torch.cuda
from tqdm import tqdm
from nli_models import StanceDetectionNliModel, iter_list
import time
from Agreement_statistics import measure_agreement, show_heatmap


def max_of_3_stances_positive(model, df, doc_batch=32):
    pro_israel = "The premise supports Israel"
    pro_hammas = "The premise supports hammas"
    neutral = "The premise supports neither Israel nor hammas"
    for hypo, prefix in zip([pro_israel, pro_hammas, neutral], ['pro_israel', 'pro_hammas', 'neutral_pro']):
        retrieve_and_rerank_predictions = []
        retrieve_and_predict_predictions = []
        for batch_docs in tqdm(iter_list(df['content'].tolist(), doc_batch)):
            retrieve_and_rerank_predictions.append(model.retrieve_and_rerank(batch_docs, hypo))
            torch.cuda.empty_cache()
            retrieve_and_predict_predictions.append(model.retrieve_and_predict(batch_docs, hypo))
            torch.cuda.empty_cache()
        retrieve_and_rerank_predictions = np.concatenate(retrieve_and_rerank_predictions, axis=0)
        retrieve_and_predict_predictions = np.concatenate(retrieve_and_predict_predictions, axis=0)
        df[f'retrieve_and_rerank_{prefix}_scores_positive'] = retrieve_and_rerank_predictions[:, 0]
        df[f'retrieve_and_rerank_{prefix}_scores_neutral'] = retrieve_and_rerank_predictions[:, 1]
        df[f'retrieve_and_rerank_{prefix}_scores_negative'] = retrieve_and_rerank_predictions[:, 2]
        df[f'retrieve_and_predict_{prefix}_scores_positive'] = retrieve_and_predict_predictions[:, 0]
        df[f'retrieve_and_predict_{prefix}_scores_neutral'] = retrieve_and_predict_predictions[:, 1]
        df[f'retrieve_and_predict_{prefix}_scores_negative'] = retrieve_and_predict_predictions[:, 2]
    return df


def max_of_3_stances_negative(model, df, doc_batch=8):
    amti_israel = "The premise is anti Israel"
    amti_hammas = "The premise is anti hammas"
    neutral = "The premise neither anti Israel nor anti hammas"
    for hypo, prefix in zip([amti_israel, amti_hammas, neutral], ['anti_israel', 'anti_hammas', 'neutral_anti']):
        retrieve_and_rerank_predictions = []
        retrieve_and_predict_predictions = []
        for batch_docs in tqdm(iter_list(df['content'].tolist(), doc_batch)):
            retrieve_and_rerank_predictions.append(model.retrieve_and_rerank(batch_docs, hypo))
            retrieve_and_predict_predictions.append(model.retrieve_and_predict(batch_docs, hypo))
        retrieve_and_rerank_predictions = np.concatenate(retrieve_and_rerank_predictions, axis=0)
        retrieve_and_predict_predictions = np.concatenate(retrieve_and_predict_predictions, axis=0)
        df[f'retrieve_and_rerank_{prefix}_scores_positive'] = retrieve_and_rerank_predictions[:, 0]
        df[f'retrieve_and_rerank_{prefix}_scores_neutral'] = retrieve_and_rerank_predictions[:, 1]
        df[f'retrieve_and_rerank_{prefix}_scores_negative'] = retrieve_and_rerank_predictions[:, 2]
        df[f'retrieve_and_predict_{prefix}_scores_positive'] = retrieve_and_predict_predictions[:, 0]
        df[f'retrieve_and_predict_{prefix}_scores_neutral'] = retrieve_and_predict_predictions[:, 1]
        df[f'retrieve_and_predict_{prefix}_scores_negative'] = retrieve_and_predict_predictions[:, 2]
    return df


def main():
    bbc = pd.read_json('data/bbc.json')
    bbc['source'] = 'bbc'
    dailymail = pd.read_json('data/dailymail.json')
    dailymail['source'] = 'dailymail'
    df = pd.concat((bbc, dailymail)).reset_index(drop=True)
    device = 'cuda:1'
    model = StanceDetectionNliModel(device=device)
    df = max_of_3_stances_positive(model, df)
    df = max_of_3_stances_negative(model, df)
    df.to_csv('data/bbc_dailymail_max_of_3_stances.csv')


def translate_results(results):
    for key in results.keys():
        data = results[key]
        labels_to_results = {'israel': 'pro israel', 'hammas': 'pro hammas', 'neutral': 'neutral'}
        labels = []
        for x in data:
            for k, v in labels_to_results.items():
                if k in x:
                    labels.append(v)
        results[key] = labels
    return results


def tranform_results_to_labels():
    df_results = pd.read_csv('data/bbc_dailymail_max_of_3_stances.csv', index_col=0)
    bbc = pd.read_json('data/bbc.json')
    bbc['source'] = 'bbc'
    dailymail = pd.read_json('data/dailymail.json')
    dailymail['source'] = 'dailymail'
    df = pd.concat((bbc, dailymail)).reset_index(drop=True)
    results = {}
    for op1 in ['pro', 'anti']:
        for op2 in ['rerank', 'retrieve']:
            if op1 == 'pro':
                df_temp = df_results[[x for x in df_results.columns if op1 in x and op2 in x and 'positive' in x]]
                cols = df_temp.columns.tolist()
                results[f"{op1}_{op2}"] = [cols[i] for i in np.argmax(df_temp.values, axis=1)]
            else:
                df_temp = df_results[[x for x in df_results.columns if op1 in x and op2 in x and 'negative' in x]]
                cols = df_temp.columns.tolist()
                results[f"{op1}_{op2}"] = [cols[i] for i in np.argmax(df_temp.values, axis=1)]
    results = translate_results(results)
    for key, value in results.items():
        df[key] = value
    df.to_csv('data/bbc_dailymail_max_of_3_stances_final_predictions.csv')


def measure_the_agreement():
    records = []
    df = pd.read_csv('data/bbc_dailymail_max_of_3_stances_final_predictions.csv', index_col=0)
    for col in ['pro_rerank', 'pro_retrieve', 'anti_rerank', 'anti_retrieve']:
        for col2 in ['pro_rerank', 'pro_retrieve', 'anti_rerank', 'anti_retrieve']:
            if col == col2:
                continue
            agreements = measure_agreement(df.copy(), col, col2)
            print(agreements)
            records.append((col, col2, agreements['Kappa'], agreements['Krippendorff  alpha'],
                            agreements['Scott pi'], agreements['Bennett s']))
            print("-------------------------------------------------------------")
    df = pd.DataFrame.from_records(records, columns=['annotations_model_1', 'annotations_model_2', 'Kappa',
                                                     'Krippendorff  alpha', 'Scott pi',
                                                     'Bennett s'])
    for metric in ['Kappa', 'Krippendorff  alpha', 'Scott pi', 'Bennett s']:
        if metric == 'Kappa':
            show_heatmap(df, 'annotations_model_1', 'annotations_model_2', metric, 0)
        else:
            show_heatmap(df, 'annotations_model_1', 'annotations_model_2', metric, -1)


if __name__ == '__main__':
    main()
    tranform_results_to_labels()
    # measure_the_agreement()
