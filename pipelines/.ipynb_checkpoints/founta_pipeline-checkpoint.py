import csv

data_path = "./data/Founta_2018/data/hatespeech_text_label_vote.csv"

data_path_2 = "./data/Founta_2018/data/crawled_data_Founta_2018.tsv"

def get_data():
    full_data = list()
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[0]
            entry['label'] = row[1]
            full_data.append(entry)
    return full_data

def get_data_binary():
    full_data = list()
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[0]
            if row[1] == "normal" or row[1]=='spam':
                entry['label'] = 'neutral'
            else:
                entry['label'] = 'abusive'
            full_data.append(entry)
    return full_data

def get_user_data():
    full_data = list()
    with open(data_path_2, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            entry = dict()
            entry['text'] = row[0]
            entry['label'] = row[-1]
            entry['id'] = row[1]
            entry['user'] = dict()
            entry['user']['id'] = row[2]
            full_data.append(entry)
    return full_data[1:]

get_data()
