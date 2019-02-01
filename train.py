import json
import scipy
import sys, os, time, random, collections, itertools
import tensorflow as tf
import model_util
import math
import numpy as np
from model import PIModel

class configuration:
    def __init__(self, attention, learning_rate, l2_norm, dropout, activation_function, vocab_dim, state_size, batch_size):
        self.retrain_embeddings = True
        self.vocab_dim = vocab_dim
        self.attention = attention
        self.max_prem_len = 9
        self.max_hyp_len = 9
        self.dropout = dropout
        self.state_size = state_size
        self.max_grad_norm = 5.
        self.batch_size = batch_size
        self.l2_norm = l2_norm
        self.learning_rate= learning_rate
        self.activationstring = activation_function


def ratio(preds, labels):
    total = 0
    right = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            right +=1
        total += 1
    return right/total

def confuse(preds, labels):
    matrix = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(preds)):
        matrix[preds[i]][labels[i]] += 1
    return matrix

def hyperparameter_search(models,  data_path, learning_rates, l2_norms, dropouts, activation_functions, vocab_dims, state_sizes, batch_sizes, num_epoch):
    embeddings = model_util.get_word_vec(vocab=model_util.get_vocab())
    word_to_id = model_util.get_word_to_id(vocab=model_util.get_vocab())
    labels = ['entailment','contradiction','neutral']
    results = dict()
    hyp_results = dict()
    for model,attention in models:
        for learning_rate in learning_rates:
            for l2_norm in l2_norms:
                for dropout in dropouts:
                    for activation_function in activation_functions:
                        for vocab_dim in vocab_dims:
                            for state_size in state_sizes:
                                for batch_size in batch_sizes:
                                    config = configuration(attention, learning_rate, l2_norm, dropout, activation_function, vocab_dim, state_size, batch_size)
                                    m = PIModel(config, embeddings, model)
                                    with tf.Session() as sess:
                                        tf.global_variables_initializer().run()
                                        params = (str(learning_rate),str(l2_norm),str(activation_function), str(dropout), str(vocab_dim), str(state_size), str(batch_size))
                                        results_folder = ""
                                        for param in params:
                                            results_folder += param
                                        if not os.path.exists(results_folder):
                                            os.mkdir(results_folder)
                                        hypaccs= []
                                        results = []
                                        best_val = 0
                                        best_test = 0
                                        count = 0
                                        for ep in range(num_epoch):
                                            print("\n>> Beginning epoch {}/{} <<".format(ep, num_epoch))
                                            ### Call training and validation routine
                                            ### This part runs one epoch of trainin#g and one epoch of validation
                                            ### Outcomes: preds, labels, constr, loss
                                            train_data = model_util.get_feed(os.path.join(data_path+  ".train"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len, shuffle = True)
                                            for prem, prem_len, hyp, hyp_len, label in train_data:
                                                _, _= m.optimize(sess, prem, prem_len, hyp, hyp_len, label)
                                                count += 1
                                                if count*config.batch_size % 100000 < config.batch_size:
                                                    val_data = model_util.get_feed(os.path.join(data_path +".val"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len)
                                                    preds_val, labels_val, _= m.run_test_epoch(sess, val_data)
                                                    train_data2 = model_util.get_feed(os.path.join(data_path+ ".train"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len, num_iter=int(10000/config.batch_size), shuffle = False)
                                                    preds_train2, labels_train2, _= m.run_test_epoch(sess, train_data2)
                                                    results.append((ratio(preds_train2, labels_train2), ratio(preds_val, labels_val),ratio(preds_test, labels_test), confuse(preds_train2, labels_train2), confuse(preds_val, labels_val),confuse(preds_test, labels_test)))
                                                    if ratio(preds_val, labels_val) > best_val:
                                                        best_val= ratio(preds_val, labels_val)
                                                        best_train= ratio(preds_train, labels_train)
                                                        saver = tf.train.Saver()
                                                        save_path = saver.save(sess, os.path.join(results_folder, "bestmodel.ckpt"))
                                                    hypaccs.append((ratio(preds_val,labels_val),ratio(preds_test, labels_test)))
                                        hyp_results[params] = best_val
                                for k in hyp_results:
                                    print(k, ":    ",hyp_results[k])
                                with open(model + "hypsearch_results", "w") as f:
                                    f.write(json.dumps())
if __name__ == '__main__':
    hyperparameter_search([("seq2seq","no")], "training_data/dummy", [0.001], [0.001], [1.0], ["tanh"], [100], [100], [64], 2)
