from preprocess import preprocess
import pandas as pd
from nltk.metrics.agreement import AnnotationTask





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
    df = pd.read_csv('data/results_baselines_llama2-7b.csv')
    df = preprocess(df)
    cols = ['llama2_processed', 'nltk_sentiment_annotations', 'nltk_sentiment_annotations_from_compound',
            'financial_sentiment_annotations', 'news_sentiment_annotations', 'political_affiliation_annotations']
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
    df = pd.DataFrame.from_records(records,columns=['annotations_model_1', 'annotations_model_2', 'Kappa', 'Krippendorff  alpha', 'Scott pi',
                       'Bennett s'])
    df.to_csv('data/agreements.csv', index=False)


if __name__ == '__main__':
    main()
