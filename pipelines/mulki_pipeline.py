import pandas as pd

def getDataFrame():
    return pd.read_csv('./data/Mulki_2017/L-HSAB.csv', sep="\t", encoding='utf-8')
    #return pd.read_excel(open('../data/Arabic_multi/Hatespeech-data-merge.xlsx', 'rb'), sheet_name='MERGE', dtype={'iD': object})  

def get_data():
    full_data = list()
    df = getDataFrame()
    for index, row in df.iterrows():
        full_data.append({'text':row['Tweet'],'label': row['Class']})
    return full_data

def get_data_binary():
    full_data = list()
    df = getDataFrame()
    for index, row in df.iterrows():
        label = 'neutral' if row['Class'] == 'normal' else 'abusive'
        full_data.append({'text':row['Tweet'],'label': label})
    return full_data