import numpy as np
import re
import pickle
import json
import matplotlib.pyplot as plt
import shap
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Activation, dot, Lambda

"""
Various helper methods for the classification pipeline and visualizations
"""

# preload word embeddings to use in an embedding layer for NNs
def construct_embedding_from_file(file_path, embedding_dim, max_features, tokenizer, encoding="utf-8"):
    embedding_file = open(file_path, encoding=encoding)

    embeddings_dictionary = dict()

    for line in embeddig_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    embedding_file.close()

    embedding_matrix = np.zeros((max_features, embedding_dim))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

# basic filtering methods
def preprocess_text(sen):
    #punct and nums
    sentence = re.sub('[^a-zA-Z#]', ' ', sen)
    #single char
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    #remove space
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

# separating inputs and labels
def get_X_y(data):
    X_tweets = list()
    y_labels = list()
    for entry in data:
        tweet = entry['text']
        tweet = preprocess_text(tweet)
        tweet = tweet.lower()
        label = entry['label']
        X_tweets.append(tweet)
        y_labels.append(label)
    return X_tweets, y_labels

# encoding into numerical format
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

# encode labels into binary format instead
def encode_binary(y_labels, non_hate_labels):
    encoding = {'non-hate': 0, 'hate': 1}
    y_encoded = list()
    for entry in y_labels:
        if entry in non_hate_labels:
            y_encoded.append(encoding['non-hate'])
        else:
            y_encoded.append(encoding['hate'])
    return y_encoded, encoding

# looks for the longest tweet in the corpus
def max_tweet_length(X_tweets):
    maxlen = 0
    for entry in X_tweets:
        en_list = entry.split()
        if len(en_list) > maxlen:
            maxlen = len(en_list)
    return maxlen

# helper to visualize train history
def plot_training_history(history):
    plt.plot(history.history['auc_1'])
    plt.plot(history.history['val_auc_1'])

    plt.title('AUC History')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.show()

# model dump to file
def save_model(model, tokenizer, model_path, tokenizer_path):
    model.save(model_path)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# printer helper to look at predictions in detail
def show_tweets_and_predictions(tokenizer, label_encoding, y_pred, y_eval, X_test, n=10):
    label_encoding = {b: a for a, b in label_encoding.items()}
    rev_map = dict(map(reversed, tokenizer.word_index.items()))
    for a,b,c in zip(y_pred, y_eval, X_test):
        if a != b:
            print("Tweet: \n%s" % " ".join([rev_map[entry] for entry in c if entry != 0]))
            print("Label: %s \nGS: %s" % (label_encoding[a], label_encoding[b]))

# self-defined 3d attention layer for Keras
def attention_3d_block(hidden_states):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    adapted from felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector

# removes attention weights for padding tokens, this only affects visualization
def set_padding_scores_to_zero(encoded_tweet, attention_scores):
    indeces = np.argwhere(encoded_tweet==0)
    attention_scores[indeces] = 0
    indeces = np.argwhere(encoded_tweet!=0)
    softmax_attn_scores =  np.exp(attention_scores[indeces])/sum(np.exp(attention_scores[indeces]))
    attention_scores[indeces] = softmax_attn_scores
    return attention_scores

# intercept attention weights from a model prediction
def get_attention_weights(model, X_test):
    intermediate_model = Model(inputs=model.input,
                                 outputs=model.get_layer(name='attention_score').output)
    attention_scores = intermediate_model.predict(X_test)
    full_attention_scores = list()
    for tweet, attn_score in zip(X_test, attention_scores):
        smax_score = set_padding_scores_to_zero(tweet, np.copy(attn_score))
        full_attention_scores.append(smax_score)
    return full_attention_scores

# ge full weights from embedding layer of a model
def get_embedding_weights(model):
    embedding_layer = model.get_layer('embedding_layer')
    embeddings = embedding_layer.get_weights()[0]
    return embeddings

# dump the attention weights with the text in a readable json format
# format predefined according to https://github.com/cbaziotis/neat-vision
def generate_json_entry(encoded_text, label, prediction, rev_map, attentions, id_digit):
    classification_object = dict()
    text = [rev_map[entry] for entry in encoded_text]
    attention_scores = [float(score) for score in attentions]
    classification_object['text'] = text
    label = int(np.argmax(label))
    classification_object['label'] = label
    prediction = int(np.argmax(prediction))
    classification_object['prediction'] = prediction
    classification_object['posterior'] = []
    classification_object['attention'] = attention_scores
    classification_object['id'] = "sample_" + str(id_digit)
    id_digit += 1
    return classification_object

# batch processing of json entries see above
def write_attention_to_json(outfile, X_test, y_test, predictions, tokenizer, full_attention_scores):
    id_digit = 1
    prediction_list = []
    rev_map = dict(map(reversed, tokenizer.word_index.items()))
    rev_map[0] = "EOT"
    for text, label, prediction, attentions in zip(X_test, y_test, predictions, full_attention_scores):
        output_object = generate_json_entry(text, label, prediction, rev_map, attentions, id_digit)
        id_digit += 1
        prediction_list.append(output_object)
    with open(outfile, 'w') as o_f:
        json.dump(prediction_list, o_f)

# generate a DeepSHAP force plot for an existing SHAP model for a specific training example
def shap_plot_example(explain_examples, explain_labels, predictions, explainer, shap_values, reverse_word_map, i):
    predicted_class = np.argmax(predictions[i])
    real_class = np.argmax(explain_labels[i])
    sent_list = list()
    for word in explain_examples[i]:
        sent_list.append(reverse_word_map[word])
    max_ind = len(sent_list)
    for ind, word in enumerate(sent_list):
        if word == "EOT":
            max_ind = ind + 1
            break
    x_test_words = " ".join(sent_list[:max_ind])
    print(x_test_words)
    shap.force_plot(explainer.expected_value[predicted_class], shap_values[predicted_class][i], np.array(sent_list), matplotlib=True)
    print("Predicted: %s - Gold Standard: %s " % (predicted_class, real_class))

# calculates scores for a confusion matrix
def perf_measure(preds, gs_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(gs_labels)):
        if gs_labels[i]==preds[i]==1:
           TP += 1
        if preds[i]==1 and gs_labels[i]!=preds[i]:
           FP += 1
        if preds[i]==gs_labels[i]==0:
           TN += 1
        if preds[i]==0 and gs_labels[i]!=preds[i]:
           FN += 1
    return (TP, FP, TN, FN)

# confusion matrix helper method, works with nxn labels
def draw_confusion_matrix(input_matrix, x_labels):
    fig, ax = plt.subplots()
    ax.matshow(input_matrix, cmap=plt.cm.Blues)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(x_labels)
    for (i, j), z in np.ndenumerate(input_matrix,):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.9', alpha=0.8))
    plt.show()
