import pandas as pd

data_training_path = "./data/Zampieri_2019/olid-training-v1.0.tsv"
data_test_path = "./data/Zampieri_2019/testset-levela.tsv"
data_test_labels_path = "./data/Zampieri_2019/labels-levela.csv"


def get_data():
    full_data = list()
    
    df_training = pd.read_csv(data_training_path,sep="\t",encoding="utf-8")
    df_test_text = pd.read_csv(data_test_path,sep="\t",encoding="utf-8")
    df_test_label = pd.read_csv(data_test_labels_path,sep=",",encoding="utf-8",names=['id','label'])
        
    df_test = pd.merge(df_test_text, df_test_label, how='inner', on='id')

    for index,row in df_training.iterrows():    
        entry = dict()
        entry['text'] = row['tweet'].replace("URL","")
        entry['label'] = row['subtask_a']
        full_data.append(entry)

    for index,row in df_test.iterrows():    
        entry = dict()
        entry['text'] = row['tweet'].replace("URL","")
        entry['label'] = row['label']
        full_data.append(entry)
        
    return full_data

def get_data_binary():
    full_data = get_data()
    for entry in full_data:
        entry['label'] = entry['label'].replace("OFF","abusive").replace("NOT","neutral")
    return full_data

def get_user_data():
    return []