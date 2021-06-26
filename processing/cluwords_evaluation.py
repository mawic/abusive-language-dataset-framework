import numpy as np
import pandas as pd
from collections import defaultdict

"""
These methods are used in CluWord Topic model evaluation.
This starts by specifying a data path where the CluWords models are saved.
"""

DATASETS_PATH = 'tools/cluwords/cluwords/multi_embedding/datasets/'

# if sampling of data sets was done this preserves our subsampling
def save_samples(samples, name):
    save_name = name +"Pre.txt"
    with open(DATASETS_PATH+save_name, 'w', encoding="utf-8") as fout:
        for tweet in samples:
            t_str = " ".join(tweet)
            fout.write(t_str + "\n")

# split tweets by data set they appear in
def separate_by_datasets(full_data, separators):
    start_index = 0
    dataframes = list()
    for separator in separators:
        dataframe = full_data.iloc[start_index:(start_index+separator)]
        dataframes.append(dataframe)
        start_index += separator
    return dataframes

# threshold based assignment of topics to a dataset
def topic_by_threshhold(dataframe, num_topics, threshold=0):
    topics = defaultdict(int)
    for i in range(num_topics):
        topics[i] = dataframe.loc[dataframe[i] > threshold].shape[0]
    # normalize
    topics = {k: v/dataframe.shape[0] for k, v in topics.items()}
    for i in range(num_topics):
        if i not in topics:
            topics[i] = 0.0
    return topics

# maximum based assignment of topics to a dataset
def topic_by_local_max(dataframe, num_topics):
    topics = defaultdict(int)
    max_cols = dataframe.idxmax(axis=1)
    for entry in max_cols:
        topics[entry] += 1
    # normalize
    topics = {k: v/dataframe.shape[0] for k, v in topics.items()}
    for i in range(num_topics):
        if i not in topics:
            topics[i] = 0.0
    return topics

# compare the prevalence of specific topics between data sets
def compare_prevalence(topic_list):
    datasets = [n for n in range(len(topic_list))]
    max_topics = dict()
    for i, topics in enumerate(topic_list):
        for topic, value in topics.items():
            if not topic in max_topics:
                max_topics[topic] = (i, value)
            else:
                if value > max_topics[topic][1]:
                    max_topics[topic] = (i, value)
    return max_topics
