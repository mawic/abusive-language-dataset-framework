import numpy as np
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from matplotlib  import cm
from collections import defaultdict

"""
Helper methods for dealing with and visualizing different embeddings and their
projections.
"""

# generate a tf-idf model for a collection of documents
def tf_idf_for_corpus(tweets):
    dct = Dictionary(tweets)
    corpus = [dct.doc2bow(tweet) for tweet in tweets]
    model = TfidfModel(corpus)
    return dct, corpus, model

# generate tweet embedding weighted by tf-idf from individual words
def generate_weighted_tweet_embedding(dct, corpus, model, w_vectors, dim):
    weighted_embedding = list()
    for tweet in corpus:
        weighted_words = list()
        tweet_tfidf = model[tweet]
        for word, tfidf in zip(tweet, tweet_tfidf):
            if dct[word[0]] in w_vectors:
                w_vector = w_vectors[dct[word[0]]]
                weighted_words.append(w_vector*tfidf[1])
            else:
                # we might not have the word in our embedding so we cant weigh it
                weighted_words.append(np.zeros(dim))
        weighted_words = np.asarray(weighted_words)
        weighted_embedding.append(weighted_words.mean(axis=0))
    weighted_embedding = np.asarray(weighted_embedding)
    return weighted_embedding

# lookup method to fetch a vector representation for a word
def get_embedding_for_word(word, embeddings, tokenizer, dim):
    if word not in tokenizer.word_index:
        return np.zeros(dim)
    w_index = tokenizer.word_index[word]
    return embeddings[w_index]

# combine word embeddings to generate tweet embedding
def get_tweet_embedding(tweet_text, embeddings, tokenizer, dim):
    # basic composition for visualization no normalization or similar
    words = tweet_text.split()
    embedding_list = list()
    for word in words:
        embedding = get_embedding_for_word(word, embeddings, tokenizer, dim)
        embedding_list.append(embedding)
    if not embedding_list:
        return np.zeros(dim)
    tweet_embedding = embedding_list[0]
    if len(embedding_list) == 1:
        return tweet_embedding
    for embedding in embedding_list[1:]:
        tweet_embedding = np.add(tweet_embedding, embedding)
    return tweet_embedding

# wrapper method to do the whole corpus in one batch
def corpus_to_embedding(tweets, embeddings, tokenizer):
    embedded_tweets = list()
    dim = embeddings.shape[1]
    for tweet in tweets:
        embedded_tweet = get_tweet_embedding(tweet, embeddings, tokenizer, dim)
        embedded_tweets.append(embedded_tweet)
    return np.stack(embedded_tweets, axis=0)

# display a tsne projected 2 dimensional embedding
def plot_tsne_embedding(tsne_embedded, labels, label_text, figsize=(18,18)):
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]

    emb_rx = np.flip(emb_x)
    emb_ry = np.flip(emb_y)

    labels.reverse()
    int_labels = labels

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x=emb_rx, y=emb_ry, s=20, c=int_labels, marker='o', cmap=cm.jet)
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles, label_text,
                        loc="upper right", title="Datasets")
    ax.add_artist(legend)

    return fig

# plot the same, but with added annotations
def plot_tsne_embedding_annotate(tsne_embedded, labels, label_text, annotation_text, figsize=(18,18)):
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]

    s_sizes = [20] * 8000 + [100] * 20 + [100] * 20
    int_labels = labels

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x=emb_x, y=emb_y, s=s_sizes, c=int_labels, edgecolors='w', marker='o', cmap=cm.jet)
    for i, text in zip(range(8000,8020), annotation_text):
        ax.annotate(text, (emb_x[i], emb_y[i]), xytext=(emb_x[i]+2, emb_y[i]+2), weight='bold', fontsize=14, bbox=dict(boxstyle='round', fc='white', ec='w', alpha=0.8))
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles, label_text,
                        loc="upper right", title="Datasets")
    ax.add_artist(legend)

# plotting method for pca embedding and annotation
def plot_embedding_annotate(tsne_embedded, labels, label_text, annotation_text, figsize=(8,8)):
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]

    int_labels = labels

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x=emb_x, y=emb_y, s=50, c=int_labels, edgecolors='w', marker='o', cmap=cm.jet)
    for i, text in enumerate(annotation_text):
        ax.annotate(text, (emb_x[i], emb_y[i]), xytext=(emb_x[i]+0.001, emb_y[i]+0.001), fontsize=12)
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles, label_text,
                        loc="upper right", title="Datasets")
    ax.add_artist(legend)
    return fig

# computes a singlar embedding per label for a data set, similar to how tweet embeddings are computed above
def transform_to_label_embed(dataset, dset_name, word_vectors, embed_dim=300):
    dataset_tag_embedding = list()
    for tweet in dataset:
        tweet_embedding = np.zeros(embed_dim)
        wcount = len(tweet['text'])
        for word in tweet['text']:
            if word in word_vectors:
                tweet_embedding += word_vectors[word]
        tweet_embedding = tweet_embedding / wcount
        label = dset_name + "_" + str(tweet['label'])
        dataset_tag_embedding.append((tweet_embedding, label))

    tag_embeddings = dict()
    tag_count = defaultdict(int)
    for entry in dataset_tag_embedding:
        if entry[1] in tag_embeddings:
            tag_embeddings[entry[1]] += entry[0]
        else:
            tag_embeddings[entry[1]] = entry[0]
        tag_count[entry[1]] += 1
    averaged_tag_embeddings = list()
    tag_labels = list()
    for key, value in tag_embeddings.items():
        averaged_tag_embeddings.append(value / tag_count[key])
        tag_labels.append(key)
    return averaged_tag_embeddings, tag_labels
