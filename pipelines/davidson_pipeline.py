import csv
import pickle

data_path = "./data/Davidson_2017/data/labeled_data.tsv"
data_user_path = "./data/Davidson_2017/crawled/davidson_with_user_timestamp_check_pickle.pkl"

def get_data():
    full_data = list()
    with open(data_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[-1]
            try:
                if int(row[-2]) == 0:
                    entry['label'] = 'hate'
                if int(row[-2]) == 1:
                    entry['label'] = 'offensive'
                if int(row[-2]) == 2:
                    entry['label'] = 'neither'
            except:
                entry['label'] = row[-2]
            full_data.append(entry)
    return full_data[1:]

def get_user_data():
    full_data = list()
    data = pickle.load(open(data_user_path, "rb"))
    for row in data:
        if 'userid_str' in row:
            entry = dict()
            entry['text'] = row['text']
            entry['label'] = row['label']
            entry['id'] = row['id_str']
            entry['user'] = dict()
            entry['user']['id'] = row['userid_str']
            full_data.append(entry)
    return full_data

def get_data_binary():
    full_data = list()
    with open(data_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[-1]
            try:
                if int(row[-2]) == 0:
                    entry['label'] = 'abusive'
                if int(row[-2]) == 1:
                    entry['label'] = 'abusive'
                if int(row[-2]) == 2:
                    entry['label'] = 'neutral'
            except:
                entry['label'] = row[-2]
            full_data.append(entry)
    return full_data[1:]
