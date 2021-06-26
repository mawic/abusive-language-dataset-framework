import json
import pickle
import csv

data_path = "../../Data/Rezvan_2018/"
appearance = data_path + "Appearance Data.csv"
intelligence = data_path + "Intelligence Data.csv"
racial = data_path + "Racial Data.csv"
political = data_path + "Political Data.csv"
sexual = data_path + "Sextual Data.csv"

def read_file(filename, label):
    read_data = list()
    with open(filename, 'r', encoding='latin-1') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            entry = dict()
            entry['text'] = row[0]
            if row[1] == 'no':
                entry['label'] = row[1]
            else:
                entry['label'] = label
            entry['dset_type'] = label
            read_data.append(entry)
    return read_data[1:]

def get_data():
    full_data = list()
    for dset, type in [(appearance, 'APP'), (intelligence, 'INT'), (racial, 'RAC'), (political, 'POL'), (sexual, 'SEX')]:
        read_data = read_file(dset, type)
        full_data.extend(read_data)
    return full_data
