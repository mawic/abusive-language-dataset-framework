import json
import pandas as pd

def loadLabels():
    test = pd.read_csv('./data/Alsafari_2020/AH-Test.csv',sep=",", encoding="utf-8", dtype={'iD': object})
    train = pd.read_csv('./data/Alsafari_2020/AH-Train.csv',sep=",", encoding="utf-8", dtype={'iD': object})
    return pd.concat([train, test])

def loadTexts():
    with open('./data/Alsafari_2020/210218_Alsafari_2020_API_dump.json') as json_file:
        data = json.load(json_file)
    return data

def get_data_binary():
    labels = dict()
    label_data = loadLabels()
    for index,row in label_data.iterrows():
        labels[str(row['ID'])] = 'neutral' if row['2-Class'] == 'C' else 'abusive'
    tweets = loadTexts()
    full_data = list()
    for elem in tweets:
        full_data.append({'text':elem['full_text'],'label': labels[elem['id_str']]})
    return full_data

def get_data():
    labels = dict()
    label_data = loadLabels()
    for index,row in label_data.iterrows():
        if row['3-Class'] == 'O':
            labels[str(row['ID'])] = 'offensive' 
        if row['3-Class'] == 'C':
            labels[str(row['ID'])] = 'clean' 
        if row['3-Class'] == 'H':
            labels[str(row['ID'])] = 'hateful' 
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
        if row['3-Class'] == 'O':
            full_data.append({'label': 'offensive','id':str(row['ID'])}) 
        if row['3-Class'] == 'C':
            full_data.append({'label': 'clean','id':str(row['ID'])}) 
        if row['3-Class'] == 'H':
            full_data.append({'label': 'hateful','id':str(row['ID'])}) 
    return full_data 

def get_available_data():
    labels = dict()
    label_data = loadLabels()
    for index,row in label_data.iterrows():
        if row['3-Class'] == 'O':
            labels[str(row['ID'])] = 'offensive' 
        if row['3-Class'] == 'C':
            labels[str(row['ID'])] = 'clean' 
        if row['3-Class'] == 'H':
            labels[str(row['ID'])] = 'hateful' 
    tweets = loadTexts()
    full_data = list()
    for elem in tweets:
        full_data.append({'text':elem['full_text'],
                          'label': labels[elem['id_str']],
                          'user': {'id':str(elem['user']['id'])},
                          'id':str(elem['id_str'])})
    return full_data
