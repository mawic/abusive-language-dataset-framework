import json
import pandas as pd

def loadLabels():
    data1 = pd.read_csv('./data/Alshalan_2020/test.csv',sep=",", encoding="utf-8", dtype={'id': object})
    data2 = pd.read_csv('./data/Alshalan_2020/train.csv',sep=",", encoding="utf-8", dtype={'id': object})

    return pd.concat([data1,data2])

def loadTexts():
    with open('./data/Alshalan_2020/210218_Alshalan_2020_API_dump.json') as json_file:
        data = json.load(json_file)
    return data

def get_data_binary():
    labels = dict()
    label_data = loadLabels()
    for index,row in label_data.iterrows():
        labels[str(row['id'])] = 'neutral' if row['class'] == 0 else 'abusive'
    tweets = loadTexts()
    full_data = list()
    for elem in tweets:
        full_data.append({'text':elem['full_text'],'label': labels[elem['id_str']]})
    return full_data

def get_data():
    labels = dict()
    label_data = loadLabels()
    for index,row in label_data.iterrows():
        if row['class'] == 1:
            labels[str(row['id'])] = 'abusive' 
        if row['class'] == 2:
            labels[str(row['id'])] = 'hateful' 
        if row['class'] == 0:
            labels[str(row['id'])] = 'normal' 
    tweets = loadTexts()
    full_data = list()
    for elem in tweets:
        full_data.append({'text':elem['full_text'],
                          'label': labels[elem['id_str']],
                          'id':str(elem['id_str'])})
    return full_data


def get_complete_data():
    label_data = loadLabels()
    full_data = list()
    for index,row in label_data.iterrows():
        if row['class'] == 1:
            full_data.append({'label': 'abusive','id':str(row['id'])}) 
        if row['class'] == 2:
            full_data.append({'label': 'hateful','id':str(row['id'])}) 
        if row['class'] == 0:
            full_data.append({'label': 'normal','id':str(row['id'])}) 
    return full_data 


def get_available_data():
    labels = dict()
    label_data = loadLabels()
    for index,row in label_data.iterrows():
        if row['class'] == 1:
            labels[str(row['id'])] = 'abusive' 
        if row['class'] == 2:
            labels[str(row['id'])] = 'hateful' 
        if row['class'] == 0:
            labels[str(row['id'])] = 'normal' 
    tweets = loadTexts()
    full_data = list()
    for elem in tweets:
        full_data.append({'text':elem['full_text'],
                          'label': labels[elem['id_str']],
                          'id':str(elem['id_str']),
                          'user': {'id':str(elem['user']['id'])}})
    return full_data