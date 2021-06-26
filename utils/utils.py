import re
import importlib
from collections import defaultdict
from datetime import datetime

"""
Various helper methods used in the analysis
"""

# to look at the data and see if the formatting works
def print_data_example(data, n=5):
    print("Ex data:")
    for entry in data[:n]:
        if 'user' in entry:
            print(entry['user']['id'])
            if 'name' in entry['user']:
                print(entry['user']['name'])
        if 'created_at' in entry:
            print(entry['created_at'])
        print(entry['text'])
        print(entry['label'])
        print("")

# simple text preprocessor
def preprocess_text(sen):
    #punct and nums
    sentence = re.sub('[^a-zA-Z#]', ' ', sen)
    #single char
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    #remove space
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

# dump part of text to a file, for instance after sampling or processing
def export_text(out_file, encoding='utf-8'):
    with open(out_file, 'w', encoding=encoding) as o_f:
        for entry in data:
            no_lb_string = entry['text'].replace('\n', ' ').replace('\r', '')
            o_f.write(no_lb_string + "\n")

# computes a datetime object for a tweet id, see
def get_tweet_timestamp(tid, v=0):
    tid = int(tid)
    offset = 1288834974657
    tstamp = (tid >> 22) + offset
    utcdttime = datetime.utcfromtimestamp(tstamp/1000)
    if v:
        print(str(tid) + " : " + str(tstamp) + " => " + str(utcdttime))
    return tstamp, utcdttime

# get the corpus sorted by labels
def separate_text_by_classes(data):
    text_corpus = defaultdict(list)
    for entry in data:
        text = entry['text']
        label = entry['label']
        text_corpus[label].append(text)
    return text_corpus

# filter tweets by specific (identity) terms to get subpopulations of the data
def find_identity_term_tweets(X_test, identity_terms, rev_map):
    identity_term_indeces = list()
    for i, entry in enumerate(X_test):
        for index in entry:
            if index != 0:
                token = rev_map[index]
                if token in identity_terms:
                    identity_term_indeces.append(i)
    return identity_term_indeces

def fetch_import_module(module_name):
    full_mod_name = "pipelines." + module_name.lower() + "_pipeline"
    mod = importlib.import_module(full_mod_name)
    return mod
