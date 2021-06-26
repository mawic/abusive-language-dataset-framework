from collections import defaultdict
import operator
import datetime
import re
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

"""
Helper methods for sampling from different data sets
"""

# get tweets and labels from data set entries
def prepare_dataset(data):
    tweets = [entry['text'].lower().split() for entry in data]
    labels = [entry['label'] for entry in data]
    sorted_tweets_labels = sorted(zip(tweets, labels), key=lambda x: x[1])
    tweets, labels = zip(*sorted_tweets_labels)
    return tweets, labels

# constructs a list of lists of tweets that share a label
def sort_tweets_and_labels(tweets, labels):
    unique_labels = list(set(labels))
    unique_indexes = [labels.index(u_label) for u_label in unique_labels]
    combined_list = [(i, l) for i, l in sorted(zip(unique_indexes, unique_labels))]
    unique_indexes = [i for i,l in combined_list]
    unique_labels = [l for i,l in combined_list]
    num_labels = len(unique_labels)
    f_label = unique_indexes[0]
    tweet_slices = list()
    for u_entry in unique_indexes[1:]:
        s_label = u_entry
        tweet_slice = tweets[f_label:s_label]
        tweet_slices.append(tweet_slice)
        f_label = s_label
    tweet_slices.append(tweets[f_label:]) # add rest of tweets
    return tweet_slices, unique_labels

# uniformly samples from all distinct classes to generate n total samples
def sample_tweets(tweets, labels, n=2000, exclude_labels=set(), on_distribution=False):
    if on_distribution:
        total_tweets = list(zip(tweets, labels))
        samples = random.sample(total_tweets, n)
        return list(zip(*samples))
    tweet_slices, unique_labels = sort_tweets_and_labels(tweets, labels)
    num_labels = len(unique_labels)
    ss_per_label = n // (num_labels - len(exclude_labels))
    chosen_samples = list()
    labels_for_sample = list()
    for i in range(num_labels):
        cur_ss_per_label = ss_per_label
        if unique_labels[i] in exclude_labels:
            continue
        if ss_per_label > len(tweet_slices[i]):
            cur_ss_per_label = len(tweet_slices[i])
        sample = random.sample(tweet_slices[i], cur_ss_per_label)
        chosen_samples.append(sample)
        labels_for_sample.append([unique_labels[i]]*cur_ss_per_label)
    if sum(len(sa) for sa in chosen_samples) < n:
        print("""Warning: Could not sample enough data.
              \tRequested sample: %i
              \tSampled: %i""" % (n, sum(len(sa) for sa in chosen_samples)))
    return ([sample for sublist in chosen_samples for sample in sublist],
           [label for sublist in labels_for_sample for label in sublist])

# writes an existing sample. Text only.
def save_combined_samples(path, sampled_data, encoding="utf-8"):
    with open(path, 'w', encoding=encoding) as fout:
        for tweet in sampled_data:
            t_str = " ".join(tweet)
            f.write(t_str + "\n")

# writes the sample together with their original labels
def save_with_labels(path, sampled_data, labels, encoding="utf-8"):
    with open(path, 'w', encoding=encoding) as fout:
        for tweet, label in zip(sampled_data, labels):
            t_str = " ".join(tweet)
            f.write(t_str + "\t" + label + "\n")

if __name__ == "__main__":
    # put in paths to all data sets to sample from if you call this directly
    # alternatively run the methods above form a notebook
    data_to_read = ["open", "data", "goes", "here"]
    n = 2000

    sampled_data = list()
    labels = list()
    for data in data_to_read:
        tweets, labels = prepare_dataset(data)
