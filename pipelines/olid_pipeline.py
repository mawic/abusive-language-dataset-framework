import json
import pickle
import csv

data_path = "../../Data/OLIDv1.0_2019/"
train = data_path + "olid-training-v1.0.tsv"
test_a = data_path + "testset-levela.tsv"
labels_a = data_path + "labels-levela.csv"
test_b = data_path + "testset-levelb.tsv"
labels_b = data_path + "labels-levelb.csv"
test_c = data_path + "testset-levelc.tsv"
labels_c = data_path + "labels-levelc.csv"

def read_train_set():
    train_data = list()
    with open(train, 'r', encoding='latin-1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[1]
            entry['label'] = row[2]
            entry['id'] = row[0]
            entry['label_target'] = row[3]
            entry['label_target_c'] = row[4]
            train_data.append(entry)
    return train_data[1:]

def read_test_set(test_set, test_data=dict()):
    with open(test_set, 'r', encoding="latin-1") as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if row[0] == 'id':
                continue
            test_data[row[0]] = {'text':row[1], 'id':row[0], 'user':None}
    return test_data

def read_labels(label_file):
    label_data = dict()
    with open(label_file, 'r', encoding="latin-1") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            label_data[row[0]] = row[1]
    return label_data

def match_test_with_labels(test_data):
    for label_file, label in [(labels_a, 'label'), (labels_b, 'label_target'), (labels_c, 'label_target_c')]:
        label_data = read_labels(label_file)
        for id, val in test_data.items():
            if id in label_data:
                val[label] = label_data[id]
    return test_data

def generate_test_standard_format(test_data):
    all_test = list()
    for id, val in test_data.items():
        all_test.append(val)
    return all_test

def read_test():
    test_data = dict()
    for test_set in [test_a, test_b, test_c]:
        test_data = read_test_set(test_set, test_data)
    test_data = match_test_with_labels(test_data)
    test_data = generate_test_standard_format(test_data)
    return test_data

def get_data():
    full_data = list()
    full_data.extend(read_train_set())
    full_data.extend(read_test())
    return full_data
