import fasttext.util
import processing.vocabulary_stats as vs
import processing.basic_stats
import processing.user_stats
import nltk
import pickle
import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import processing.emoji as emoji
import processing.preprocessing_multilingual as preprocessing_multilingual

from gensim import corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.manifold import TSNE
from utils import embedding_utils, dataset_sampling
from utils.utils import print_data_example, preprocess_text, get_tweet_timestamp, separate_text_by_classes
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from matplotlib  import cm
from time import gmtime, strftime
from math import log

nltk.download('stopwords')

def encode_labels(y_labels):
    encoding = dict()
    max_label = 0
    y_encoded = list()
    for entry in y_labels:
        if entry not in encoding:
            encoding[entry] = max_label
            max_label += 1
        y_encoded.append(encoding[entry])
    return y_encoded, encoding
    

def transform_to_embed(dataset, dset_name,word_vectors):
    dataset_tag_embedding = list()
    for tweet in dataset:
        tweet_embedding = np.zeros(300)
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
    

def transform_to_embed_fasttext(dataset, dset_name,fasttext):
    dataset_tag_embedding = list()
    twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
    hashtag_re = re.compile(r'\B(\#[a-zA-Z0-9]+\b)(?!;)')
    for tweet in dataset:
        tweet_embedding = np.zeros(300)
        wcount = len(tweet['text'])
        for word in tweet['text']:
            #word = twitter_username_re.sub("",word)
            #word = hashtag_re.sub("",word)
            tweet_embedding += fasttext.get_word_vector(word)
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
    

def transform_to_embed_sentence_fasttext(dataset, dset_name,fasttext):
    dataset_tag_embedding = list()

    # get sentence vector
    for tweet in dataset:
        if not isinstance(tweet['text'], str):
            continue
        text = preprocessing_multilingual.clean_text(tweet['text'])
        tweet_embedding = fasttext.get_sentence_vector(text.strip())

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
        
    # average vectors
    averaged_tag_embeddings = list()
    tag_labels = list()
    for key, value in tag_embeddings.items():
        averaged_tag_embeddings.append(value / tag_count[key])
        tag_labels.append(key)
    return averaged_tag_embeddings, tag_labels


def getInterClassSimilarityMultiple(datasets,dataset_names,embedding_path,filter_classes=None):
    # load embedding
    #word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    word_vectors = fasttext.load_model(embedding_path)
    
    dataset_embeddings = []
    dataset_labels = []
    averaged_tag_embeddings = []
    tag_labels = []
    labels_count = []
    for i,dataset in enumerate(datasets):
        #embedding, labels = transform_to_embed(dataset, dataset_names[i],word_vectors)
        embedding, labels = transform_to_embed_sentence_fasttext(dataset, dataset_names[i],word_vectors)
        dataset_embeddings.append(embedding)
        dataset_labels.append(labels)
        labels_count.append(len(labels))
        averaged_tag_embeddings += embedding
        tag_labels +=labels
    
    n_averaged_tag_embeddings = np.nan_to_num(averaged_tag_embeddings)
    
    # filter classes
    #if filter_classes is not None:
    #    to_delete = []
    #    for i,class_elem in enumerate(list(tag_labels)):
    #        if class_elem not in filter_classes:
    #            to_delete.append(i)
    #    n_averaged_tag_embeddings = np.delete(n_averaged_tag_embeddings, to_delete, 0)
    #    tag_labels = np.delete(tag_labels, to_delete, 0)
    
    # set labels
    y_encode, label_encoding = encode_labels(tag_labels)
    label_text = label_encoding.keys()
    
    # apply PCA
    pca = PCA(n_components=2)
    pca_tag_embeddings = pca.fit_transform(n_averaged_tag_embeddings)
    
    # plot PCA results
    fig = plot_embedding_annotate(pca_tag_embeddings, y_encode, list(tag_labels), list(tag_labels),labels_count,pca.explained_variance_ratio_,dataset_names)
    
    
def plot_embedding_annotate(tsne_embedded, labels, label_text, annotation_text,labels_count,axis_labels,dataset_names,palette = "colorblind"):
    path_fig = "./results/"+strftime("%Y%m%d", gmtime())+ "-" + "-".join(dataset_names).replace(" ","_")
    colors = sns.color_palette(palette, len(labels_count))
    sns.set_palette(palette, len(labels_count))
    
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]
    
    int_labels = labels

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    offset = 0
    texts = []
    for i, number_of_labels in enumerate(labels_count):
        start = offset
        end = offset + number_of_labels
        scatter = ax.scatter(x=emb_x[start:end], y=emb_y[start:end], s=50, edgecolors='w', marker='o',c=[colors[i]])
        for j in range(start,end):
            texts.append(plt.text(emb_x[j], emb_y[j], annotation_text[j],fontdict={'color':colors[i],'size': 12}))
        offset = end
        

    iterations = adjust_text(texts,lim=2000,arrowprops=dict(arrowstyle='-', color='grey'))
    print(iterations)
    handles, labels = scatter.legend_elements()
    ax.set_xlabel('standardized PC1 ({:.2%} explained var.)'.format(axis_labels[0]))
    ax.set_ylabel('standardized PC2 ({:.2%} explained var.)'.format(axis_labels[1]))
    
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.png", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.eps", bbox_inches='tight', dpi=600)
    return fig


def getSimilarityScores(data):
    class_separated_dicts = separate_text_by_classes(data)
    labels = list(class_separated_dicts.keys())
    total_corpus = list()
    for label in labels:
        total_corpus.append(processing.basic_stats.filter_corpus(class_separated_dicts[label]))
        
    ### CLASS SIMILARITIES
    dictionary = corpora.Dictionary([item for sublist in total_corpus for item in sublist])
    corpus = [dictionary.doc2bow(text) for text in [item for sublist in total_corpus for item in sublist]]
    #LSI similarity
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=16)
    index = similarities.MatrixSimilarity(lsi[corpus])
    #index.save('waseem_hatespeech.index')

    similarity_scores = dict()
    for i, entry in enumerate(labels):
        #print(i,labels)
        scores = processing.basic_stats.get_total_sim_score(i, total_corpus, labels, dictionary, lsi, index)
        similarity_scores[entry] = scores
        #print(scores)
    return similarity_scores



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




def plotIntraDatasetSimilarityMultiple(title,subtitles,datasets,rows=2,cols=1,sync_scaling=False,cmap = "Blues",width=12,height=6):
    # path for storing image
    path_fig = "./results/"+strftime("%Y%m%d", gmtime())+ "-" + "-".join(subtitles).replace(" ","_")
    
    fig2 = plt.figure(constrained_layout=True,figsize=(width, height))
    fig2.suptitle(title,y=1.05,fontsize=16)
    spec2 = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig2)
    
    global_min = 1
    global_max = 0
    all_similarity_scores = []
    for dataset in datasets:
        scores = getSimilarityScores(dataset)
        all_similarity_scores.append(scores)
        for entry in scores.values():
            for ef in entry.values():
                global_min = min(global_min,ef)
                global_max = max(global_max,ef)

    m = 0
    ax = []
    for k in range(rows):
        for l in range(cols):
            if m < len(datasets):
                t_list = list()
                for entry in all_similarity_scores[m].values():
                    a_list = list()
                    for ef in entry.values():
                        a_list.append(ef)
                    t_list.append(a_list)

                ax.append(fig2.add_subplot(spec2[k, l]))
                #ax.append(fig2.add_subplot(spec2[k, l]))
                # define colors and style
                labels = all_similarity_scores[m].keys() 
            
                sheat = sns.heatmap(t_list, annot=True, fmt=".2f",
                                    ax=ax[m],square=False, label=subtitles[m], 
                                    xticklabels=labels, yticklabels=labels,
                                    vmin=global_min, vmax=global_max,
                                    cmap=cmap,cbar=False)
                sheat.set_xticklabels(labels, rotation=45, ha='center')
                sheat.set_yticklabels(labels, rotation=0, ha='right')
                #ax[m].xaxis.set_label_text(subtitles[m])
                ax[m].set_title(subtitles[m])
                #ax.set_title(subtitles[m])
                
                m += 1
    fig2.savefig(path_fig + "-vocab_intra-dataset-sim.pdf", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-vocab_intra-dataset-sim.png", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-vocab_intra-dataset-sim.eps", bbox_inches='tight', dpi=600)

def getMatrix(X):
    cv = CountVectorizer(max_df=0.95, 
                         min_df=2,                 
                         max_features=10000, 
                         binary=True)
    
    X_vec = cv.fit_transform(X)
    words = cv.get_feature_names()
    return X_vec,words

def getPmisPerClass(X_vec,Y,words):
    # get labels
    labels = set(Y)
    # create empty dict for results
    pmis_per_class = dict()
    for label in labels:
        pmis_per_class[label] = dict()

    X_matrix = np.array(X_vec.toarray())
    Y = np.array(Y)
    for i in range(len(X_matrix[0,:])):
        pmis = []
        for label in labels:
            p_label = np.sum(Y == label) / len(Y)
            select_y = Y == label
            column = X_matrix[:,i]
            select_column = column[select_y]
            p_label_x = np.sum(select_column) / len(select_column)
            if p_label_x <= 0:
                pass
                #pmis_per_class[label][words[i]] = 0
            else:
                pmi = log((p_label_x/p_label))
                pmis_per_class[label][words[i]] = pmi
    return pmis_per_class

def getTopWordOfClasses(data_sets_text,dataset_names,exclude, n=10, language='english'):
    content_table = dict()
    for dataset,dataset_name,exclude_labels in zip(data_sets_text,dataset_names,exclude):
        X = [preprocessing_multilingual.preprocess_text(x['text'],language=language) for x in dataset]
        Y = [x['label'] for x in dataset]

        X_vec,words = getMatrix(X)

        pmis_per_class = getPmisPerClass(X_vec,Y,words)
        for label in pmis_per_class.keys():
            if label in exclude_labels:
                continue
            sorted_dict = dict(sorted(pmis_per_class[label].items(), key=lambda item: item[1],reverse=True))
            column = []
            for i,word in enumerate(sorted_dict.keys()):
                if i >= n:
                    break
                column.append(word)

            content_table[dataset_name + " - " + label] = column
    return pd.DataFrame.from_dict(content_table)