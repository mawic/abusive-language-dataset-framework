import pandas as pd
import datasets
from datasets import Dataset

def filter_columns(dataset, column_names=['text','label']):
    full_data = list()
    for data in dataset:
        entry = {}
        for column in column_names:
            entry[column] = data[column]
        full_data.append(entry)
    return full_data

def convertListToDataset(datset):
    df = pd.DataFrame(datset)
    labels = list(set(df['label'].tolist()))
    ds = Dataset.from_pandas(df) 
    return ds

def get_huggingface_dataset_format(data):
    return convertListToDataset(filter_columns(data))