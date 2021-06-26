import json
import pickle
import csv

"""
Fixes our data imports for the already downloaded Waseem json in Mishras format
"""
data_path = "./data/Charitidis_2019/ENG/"
d_hate = data_path + "Train_Hate.p"
d_p_attack = data_path + "Train_Personal_Attack.p"

l_hate = data_path + "Train_Hate.csv"
l_p_attack = data_path + "Train_Personal_Attack.csv"

def get_pickle_data():
    full_data = list()
    for data in [d_hate, d_p_attack]:
        with open(data, 'rb') as f:
            f_data = pickle.load(f)
            full_data.append(f_data)
    return full_data

def get_label_data(data_source):
    labels = dict()
    with open(data_source, 'r') as f:
        csv_file = csv.reader(f)
        for n, row in enumerate(csv_file):
            if n == 0:
                continue
            labels[int(float(row[0]))] = int(row[1])
    return labels

def get_data():
    pickled_data = get_pickle_data()
    hate_labels = get_label_data(l_hate)
    p_attack_labels = get_label_data(l_p_attack)
    pickled_hate = pickled_data[0]
    pickled_attack = pickled_data[1]

    full_data = list()
    for entry in pickled_hate:
        if hasattr(entry, 'text'):
            entry_obj = dict()
            entry_obj['text'] = entry.text
            if hate_labels[entry.id] == 1:
                entry_obj['label'] = 1
            else:
                entry_obj['label'] = 0
            entry_obj['id'] = entry.id
            user_obj = dict()
            user_obj['id'] = entry.user.id
            user_obj['created_at'] = entry.user.created_at
            user_obj['verified'] = entry.user.verified
            entry_obj['user'] = user_obj
            full_data.append(entry_obj)
    for entry in pickled_attack:
        if hasattr(entry, 'text'):
            entry_obj = dict()
            entry_obj['text'] = entry.text
            if p_attack_labels[entry.id] == 1:
                entry_obj['label'] = 2
            else:
                entry_obj['label'] = 0
            entry_obj['id'] = entry.id
            user_obj = dict()
            user_obj['id'] = entry.user.id
            user_obj['created_at'] = entry.user.created_at
            user_obj['verified'] = entry.user.verified
            entry_obj['user'] = user_obj
            full_data.append(entry_obj)
    return full_data

def get_data_binary():
    pickled_data = get_pickle_data()
    hate_labels = get_label_data(l_hate)
    p_attack_labels = get_label_data(l_p_attack)
    pickled_hate = pickled_data[0]
    pickled_attack = pickled_data[1]

    full_data = list()
    for entry in pickled_hate:
        if hasattr(entry, 'text'):
            entry_obj = dict()
            entry_obj['text'] = entry.text
            if hate_labels[entry.id] == 1:
                entry_obj['label'] = 'abusive'
            else:
                entry_obj['label'] = 'neutral'
            entry_obj['id'] = entry.id
            user_obj = dict()
            user_obj['id'] = entry.user.id
            user_obj['created_at'] = entry.user.created_at
            user_obj['verified'] = entry.user.verified
            entry_obj['user'] = user_obj
            full_data.append(entry_obj)
    for entry in pickled_attack:
        if hasattr(entry, 'text'):
            entry_obj = dict()
            entry_obj['text'] = entry.text
            if p_attack_labels[entry.id] == 1:
                entry_obj['label'] = 'abusive'
            else:
                entry_obj['label'] = 'neutral'
            entry_obj['id'] = entry.id
            user_obj = dict()
            user_obj['id'] = entry.user.id
            user_obj['created_at'] = entry.user.created_at
            user_obj['verified'] = entry.user.verified
            entry_obj['user'] = user_obj
            full_data.append(entry_obj)
    return full_data
