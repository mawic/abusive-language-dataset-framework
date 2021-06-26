import pandas as pd

def getDataFrame():
    return pd.read_excel(open('./data/Arabic_multi/Hatespeech-data-merge.xlsx', 'rb'), sheet_name='MERGE', dtype={'iD': object})  

def preprocess(text):
    text = text.replace('.IDX','')
    text = text.replace('<url>','')
    return text

def get_data():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'shammur'].iterrows():
        if not isinstance(row['text'], str):
            continue
        if row['label (HS)'] == 'Non-Offensive':
            label = 'neutral' 
        elif row['label2'] == '-':
            label = 'offensive'
        elif row['label2'] == 'HS':
            label = 'hate_speech'
        elif row['label2'] == 'V':
            label = 'vulgar'
        full_data.append({'text':preprocess(row['text']),'label': label})
    return full_data

def get_data_binary():
    full_data = list()
    df = getDataFrame()
    for index, row in df[df['dataset'] == 'shammur'].iterrows():
        if not isinstance(row['text'], str):
            continue
        label = 'neutral' if row['label (HS)'] != 'Offensive' else 'abusive'
        full_data.append({'text':preprocess(row['text']),'label': label})
    return full_data