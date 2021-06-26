import pandas as pd

def getDataFrame():
    return pd.read_excel(open('./data/Arabic_multi/Hatespeech-data-merge.xlsx', 'rb'), sheet_name='MERGE', dtype={'iD': object})  

def get_data():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'osact2020'].iterrows():
        if not isinstance(row['text'], str):
            continue
        if row['label (HS)'] == 'HS':
            label = 'hate_speech' 
        elif row['label2'] == 'OFF':
            label = 'offensive'
        else:
            label = 'neutral'
        full_data.append({'text':row['text'].replace("<LF>"," ").replace("URL"," "),'label': label})
    return full_data

def get_data_binary():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'osact2020'].iterrows():
        if not isinstance(row['text'], str):
            continue
        label = 'neutral' if row['label2'] == 'NOT_OFF' else 'abusive'
        full_data.append({'text':row['text'].replace("<LF>"," ").replace("URL"," "),'label': label})
    return full_data