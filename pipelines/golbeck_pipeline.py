import json
import pickle
import csv

data_path = "../../data/Golbeck_2017/onlineHarassmentDataset.tdf"

def get_data():
    full_data = list()
    with open(data_path, 'r', encoding='latin-1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[2]
            entry['label'] = row[1]
            entry['id'] = row[0]
            full_data.append(entry)
    return full_data[1:]

def get_data_binary():
    full_data = list()
    with open(data_path, 'r', encoding='latin-1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[2]
            if row[1]=='N':
                entry['label'] = 'neutral'
            else:
                entry['label'] = 'abusive'
            entry['id'] = row[0]
            full_data.append(entry)
    return full_data[1:]
