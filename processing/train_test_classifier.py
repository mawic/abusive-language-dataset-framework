import torch
import numpy as np
import pandas as pd
import datasets
import logging
import scipy as sp
import gc
import pickle
from numba import cuda

from tqdm import tqdm
from datasets import concatenate_datasets,Dataset,ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, Trainer, TrainingArguments

import argparse
import os

def convertLabelsToInt(dataset):
    label_to_int = {
        "neutral": 0,
        "abusive": 1
    }
    dataset = dataset.map(lambda convertLabels: {"label": label_to_int[convertLabels["label"]]})
    return dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(i,model_name, number_tokens,batch,training_sets, validation_sets, test_sets, path_model,path_output,path_log,epochs):
    log = str(i) + ". Model"
    logging.warning('#'*50)
    logging.warning(log)
    logging.warning('#'*50)

    
    # define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name,num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
    
    # tokenize datasets
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)
    
    train_dataset = training_sets[i].map(tokenize, batched=True, batch_size=len(training_sets[i]))
    val_dataset = validation_sets[i].map(tokenize, batched=True, batch_size=len(training_sets[i]))
    #test_dataset = test_sets[i].map(tokenize, batched=True, batch_size=len(training_sets[i]))

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    #test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    for elem in train_dataset:
        print(len(elem['input_ids']))
    
    log = "Length of training set: " + str(len(train_dataset))
    logging.warning(log)
    
    # define trainer
    training_args = TrainingArguments(
        output_dir=path_model,          # output directory
        num_train_epochs=epochs,              # total # of training epochs
        per_device_train_batch_size=batch,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=path_log,            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics
    )
    
    # train model
    trainer.train()
    
    # evaluate model
    
    f1 = []
    precision = []
    recall = []
    #predictions = []
        
    for j in range(len(test_sets)):
        # prepare evluation test set
        eval_dataset = test_sets[j].map(tokenize, batched=True, batch_size=len(train_dataset))
        eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # predict
        results_it = trainer.predict(eval_dataset) 
        
        f1.append(results_it.metrics['eval_f1'])
        precision.append(results_it.metrics['eval_precision'])
        recall.append(results_it.metrics['eval_recall'])
        #predictions.append(results)
        
    results = {}
    results['f1'] = f1
    results['precision'] = precision
    results['recall'] = recall
    #results['predictions'] = predictions
        
    # save model
    trainer.save_model(path_model)
        
        
    pickle.dump(results, open(path_output, "wb"))
    
    
    # clear GPU memory
    del model, trainer, tokenizer
    gc.collect()
    cuda.select_device(0)
    cuda.close()
    
def main():
    
    parser = argparse.ArgumentParser(description='List the content of a folder')

    parser.add_argument('-i','--counter', type=int,
                        help='counter')
    parser.add_argument('-b','--batch', type=int,
                        help='batch size')
    parser.add_argument('-t','--training',
                        help='training sets')
    parser.add_argument('-v','--validation',
                        help='validations sets')
    parser.add_argument('-e','--test',
                        help='test sets')
    parser.add_argument('-l','--length', type=int,
                        help='token length')
    parser.add_argument('-o','--output',
                        help='output')
    parser.add_argument('-m','--model',
                        help='model')
    parser.add_argument('-g','--logs',
                        help='log')
    parser.add_argument('-c','--epochs', type=int,
                        help='log')
    parser.add_argument('-p','--path',
                        help='path')

    args = parser.parse_args()
    
    input_counter = args.counter
    input_training = args.training
    input_validation = args.validation
    input_test = args.test
    input_batch = args.batch
    input_model = args.model
    input_token_length = args.length
    input_output_path = args.output
    input_tmp_path = args.path
    input_log_path = args.logs
    input_epochs = args.epochs
            
    paths_training = input_training.split(',')
    paths_validation = input_validation.split(',')
    paths_test = input_test.split(',')
    
    training_sets = []
    validation_sets = [] 
    test_sets = []
    
    #load data
    for dataset in paths_training:
        training_sets.append(datasets.load_from_disk(dataset))
    for dataset in paths_validation:
        validation_sets.append(datasets.load_from_disk(dataset))
    for dataset in paths_test:
        test_sets.append(datasets.load_from_disk(dataset))

    # train and test model
    train_model(input_counter,input_model,input_token_length,input_batch,
                training_sets, validation_sets, test_sets,
                input_tmp_path, input_output_path, input_log_path,input_epochs)
    
    
if __name__ == "__main__":
    
    main()