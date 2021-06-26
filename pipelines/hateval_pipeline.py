import json
import pickle
import csv

data_path = "../../Data/HatEval/raw_data/"
dev = data_path + "public_development_en/dev_en.tsv"
test = data_path + "reference_test_en/en.tsv"
train = data_path + "public_development_en/train_en.tsv"

def read_file(filename):
    read_data = list()
    with open(filename, 'r', encoding='latin-1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[1]
            entry['label'] = row[2]
            entry['id'] = row[0]
            entry['label_aggressive'] = row[3]
            entry['label_target'] = row[4]
            read_data.append(entry)
    return read_data[1:]

def get_data():
    full_data = list()
    for dset in [train, dev, test]:
        read_data = read_file(dset)
        full_data.extend(read_data)
    return full_data
