from nltk.corpus import stopwords
from collections import defaultdict
from datetime import datetime, date
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
import re
import matplotlib.pyplot as plt

register_matplotlib_converters()

"""
These are helper methods for the basic statistics computed.
They are generalized to work with the standard formatting of tweets and labels.
"""

# generates a basic frequency dictionary of terms
def get_vocabulary(data):
    vocab = defaultdict(int)
    stop_words = set(stopwords.words('english'))
    for entry in data:
        tokens = re.split(r"[;,\'\s]", entry['text'])
        for token in tokens:
            if token not in stop_words:
                vocab[token.lower()] += 1
    return vocab

# Cleaning up raw input text
def filter_corpus(corpus_input):
    stop_words = set(stopwords.words('english'))
    filtered_corpus = [
                [token for token in document.lower().split() if token not in stop_words]
                for document in corpus_input
    ]
    corp_frequency = defaultdict(int)
    for text in filtered_corpus:
        for token in text:
            corp_frequency[token] += 1
    filtered_corpus = [
                        [token for token in text if corp_frequency[token] > 1]
                        for text in filtered_corpus
    ]
    return filtered_corpus

# activity for each registered day in a data set
def get_activity(timestamps):
    """
    Activity count
    """
    active_days = defaultdict(int)
    for timestamp in timestamps:
        if isinstance(timestamp, datetime):
            save_timestamp = timestamp.date()
        else:
            save_timestamp = timestamp[:11] + timestamp[-4:]
        active_days[save_timestamp] += 1
    return active_days

# activity on a monthly basis
def get_activity_months(timestamps):
    active_months = defaultdict(int)
    for timestamp in timestamps:
        if isinstance(timestamp, datetime):
            save_timestamp = timestamp.strftime('%b %Y')
        else:
            save_timestamp = timestamp[4:8] + timestamp[-4:]
        active_months[save_timestamp] += 1
    return active_months

# formatting of activity data for nicer handling by matplotlib
def activity_for_print(activity, monthly=True):
    dates = list()
    counts = list()
    for datum, count in activity.items():
        if monthly:
            date_object = datetime.strptime(datum, '%b %Y')
        elif isinstance(datum, date):
            date_object = datum
        else:
            date_object = datetime.strptime(datum, '%a %b %d %Y')
        dates.append(date_object)
        counts.append(count)
    dates_sort, counts_sort = (list(t) for t in zip(*sorted(zip(dates, counts))))
    return dates_sort, counts_sort

# show daily or monthly activity, adjust the size by fsize if the plot is too cramped or wide
def plot_activity(activity, monthly=True, fsize=None, unit='M'):
    if not fsize:
        if monthly:
            fsize = (13,1)
            unit = 'M'
        else:
            fsize = (200,1)
            unit = 'D'
    dates_sort, counts_sort = activity_for_print(activity, monthly)
    X = pd.to_datetime(dates_sort)
    fig, ax = plt.subplots(figsize=fsize)
    plt_date = ax.scatter(X, [1]*len(X), c=counts_sort,
                          marker='s', s=2000)
    fig.autofmt_xdate()

    ax.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    ax.get_yaxis().set_ticklabels([])
    day = pd.to_timedelta(1, unit=unit)
    plt.xlim(X[0] - day, X[-1] + day)
    plt.colorbar(plt_date, aspect=4)
    plt.show()

# basic similarity scoring by cosine distance
def get_sim_score(query_i, corpus, labels, dictionary, lsi, index):
    query = dictionary.doc2bow(query_i)
    vec_lsi = lsi[query]
    sims = index[vec_lsi]
    all_scored = dict()
    index = 0
    for i, entries in enumerate(corpus):
        similarity_score = sum(sims[index:index+len(entries)])/len(entries)       
        tag = labels[i]
        all_scored[tag] = similarity_score
        index += len(entries)
    return all_scored

# summing over similarities
def get_total_sim_score(query, corpus, labels, dictionary, lsi, index):
    query_corpus = corpus[query]
    tally_scores = defaultdict(float)
    for entry in query_corpus:
        scores = get_sim_score(entry, corpus, labels, dictionary, lsi, index)
        for label, score in scores.items():
            tally_scores[label] += score
    for label, score in tally_scores.items():
        tally_scores[label] = score/len(query_corpus)
    return tally_scores

# display labels and counts
def show_class_distribution(labels, corpus):
    #class dist chart
    sizes = list()
    for entry in corpus:
        sizes.append(len(entry))
    fig1, ax1 = plt.subplots()
    ax1.bar(labels, sizes)

    plt.show()

# similarity plot as a label x label matrix
def plot_similarity_scores(similarity_scores):
    t_list = list()
    for entry in similarity_scores.values():
        a_list = list()
        for ef in entry.values():
            a_list.append(ef)
        t_list.append(a_list)
    total_sim = np.asarray(t_list)
    x_labels = [0] + list(similarity_scores.keys())
    fig, ax = plt.subplots()
    ax.matshow(total_sim, cmap=plt.cm.plasma)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(x_labels)
    for (i, j), z in np.ndenumerate(total_sim,):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.9', alpha=0.8))
    plt.show()
