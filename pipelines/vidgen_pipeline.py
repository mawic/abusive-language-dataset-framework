import pandas as pd


data_path = PATH_EXPORT_PKL = "./data/Vidgen_2020/1_crawled/hs_AsianPrejudice_20kdataset_cleaned_with_user_ids.pkl"

data_annotations =  "./data/Vidgen_2020/1_crawled/hs_AsianPrejudice_40kdataset_cleaned_anonymized.tsv"


def shortenLabel(label):
    if label == "none_of_the_above":
        return "none"
    if label == "entity_directed_hostility":
        return "hostility"
    if label == "discussion_of_eastasian_prejudice":
        return "prejudice"
    if label == "entity_directed_criticism":
        return "cristism"
    if label == "counter_speech":
        return "counter"

def get_data():
    full_data = list()
    df = pd.read_pickle(data_path)
    for index,row in df.iterrows():
        entry = dict()
        entry['text'] = row['text']
        entry['label'] = shortenLabel(row['expert'])
        full_data.append(entry)
    return full_data

def get_data_binary():
    full_data = list()
    df = pd.read_pickle(data_path)
    for index,row in df.iterrows():
        entry = dict()
        entry['text'] = row['text']
        if row['expert']== "entity_directed_hostility":
            entry['label'] = 'abusive'
        else:
            entry['label'] = 'neutral'
        full_data.append(entry)
    return full_data

def get_user_data():
    full_data = list()
    df = pd.read_pickle(data_path)
    for index,row in df.iterrows():
        if row['user_id'] != '':
            entry = dict()
            entry['text'] = row['text']
            entry['label'] = shortenLabel(row['expert'])
            entry['id'] = row['tweet_id']
            entry['user'] = dict()
            entry['user']['id'] = row['user_id']
            full_data.append(entry)
    return full_data