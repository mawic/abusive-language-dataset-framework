import pandas as pd

def getDataFrame():
    return pd.read_excel(open('../data/Arabic_multi/Hatespeech-data-merge.xlsx', 'rb'), sheet_name='MERGE', dtype={'iD': object})  

def get_data():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'raghadsh'].iterrows():
        label = 'neutral' if row['label (HS)'] == 'NOT_HS' else 'hate_speech'
        full_data.append({'text':row['text'],'label': label})
    return full_data

def get_data_binary():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'raghadsh'].iterrows():
        label = 'neutral' if row['label (HS)'] == 'NOT_HS' else 'abusive'
        full_data.append({'text':row['text'],'label': label})
    return full_data

def get_user_data():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'raghadsh'].iterrows():
        label = 'neutral' if row['label (HS)'] == 'NOT_HS' else 'abusive'
        full_data.append({'text':row['text'],'label': label,'id':row['iD']})
    return full_data