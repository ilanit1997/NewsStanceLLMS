import os

from preprocess import preprocess_df
import pandas as pd
from nltk.metrics.agreement import AnnotationTask
import seaborn as sns
import matplotlib.pyplot as plt
from constants import DataCols, ParsedDataCols, ProClasses


def measure_agreement(df, col1, col2):
    column_1 = df.copy()[[col1]]
    column_1['coder'] = 1
    column_1 = column_1[['coder', col1]]
    column_1 = [tuple(x) for x in column_1.to_records()]
    column_1 = [(b, a, c) for (a, b, c) in column_1]
    # column_1 = [x for x in column_1]
    column_2 = df.copy()[[col2]]
    column_2['coder'] = 2
    column_2 = column_2[['coder', col2]]
    column_2 = [tuple(x) for x in column_2.to_records()]
    column_2 = [(b, a, c) for (a, b, c) in column_2]
    data = column_1 + column_2
    # data = [tuple(item)[1:] for item in data]
    annotation = AnnotationTask(data=data)
    return {"Kappa": annotation.kappa(), "Krippendorff  alpha": annotation.alpha(), "Scott pi": annotation.pi(),
            "Bennett s": annotation.S()}


def main():
    latest_date = "20231031"
    df = pd.read_csv(f'output/{latest_date}/results_baselines_llama2-13b.csv')
    df = preprocess_df(df)
    cols = [ParsedDataCols.LLAMA2_AFFILIATION.value, ParsedDataCols.NLTK_BEST.value, DataCols.NLTK_SENTIMENT_FROM_COMPOUND.value,
            DataCols.FINANCIAL_SENTIMENT.value, DataCols.NEWS_SENTIMENT.value, DataCols.POLITICAL_AFFILIATION.value]
    records = []
    for col in cols:
        for col2 in cols:
            if col == col2:
                continue
            try:
                agreements = measure_agreement(df.copy(), col, col2)
                print(agreements)
                records.append((col, col2, agreements['Kappa'], agreements['Krippendorff  alpha'],
                                agreements['Scott pi'], agreements['Bennett s']))
            except Exception as error:
                print(error)
                print(col)
                print(col2)
                records.append((col, col2, None, None, None, None))
            print("-------------------------------------------------------------")
    df = pd.DataFrame.from_records(records, columns=['annotations_model_1', 'annotations_model_2', 'Kappa',
                                                     'Krippendorff  alpha', 'Scott pi',
                                                     'Bennett s'])
    df.to_csv(f'output/{latest_date}/agreements.csv', index=False)
    for metric in ['Kappa', 'Krippendorff  alpha', 'Scott pi', 'Bennett s']:
        if metric == 'Kappa':
            show_heatmap(df, metric, 0, latest_date)
        else:
            show_heatmap(df, metric,-1, latest_date)


def show_heatmap(df, col, vmin, latest_date):
    plots_folder = f"output/{latest_date}/plots"
    os.makedirs(plots_folder, exist_ok=True)
    plt.figure(constrained_layout=True)
    df = df.copy()[['annotations_model_1', 'annotations_model_2', col]]
    df_pivot = df.pivot(index='annotations_model_1', columns='annotations_model_2', values=col)
    sns.heatmap(df_pivot, annot=True, vmin=vmin, vmax=1, cmap='coolwarm')
    plt.title(f"Annotation {col}")
    fig_file = f'{plots_folder}/Agreement Annotation {col}.png'
    plt.savefig(fig_file)
    plt.show()


if __name__ == '__main__':
    main()
