import json
import scipy
import sys, os, time, random, collections, itertools
import tensorflow as tf
import model_util
import math
import numpy as np
from model import PIModel

class config:
    vocab_dim = 100
    retrain_embeddings = True
    attention = "wordbyword"
    max_prem_len = 22
    max_hyp_len = 22
    dropout = 1.0
    state_size = 100
    max_grad_norm = 5.
    batch_size = 64
    l2 = 0.000
    lr = 0.001
    num_epoch = 3
    data_path = "training_data"
    activationstring = "tanh"

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

if __name__ == '__main__':
    experiment = "experiment1_level2"
    pretrained_embeddings = model_util.get_word_vec(vocab=model_util.get_vocab())
    word_to_id = model_util.get_word_to_id(vocab=model_util.get_vocab())
    m = PIModel(config, pretrained_embeddings, "seq2seq")
    labels = ['entails','contradicts','permits']
    cat_names = ['{}=>{}'.format(x,y) for x,y in itertools.product(labels,labels)]
    results = dict()
    print("\n-----\n")
    print("Learning Rate:", config.lr)
    print("Dropout:", config.dropout)
    print("L2:", config.l2)
    hyp_results = dict()
    for lr in [0.001, 0.0001,0.0003]:
        for l2 in [0,0.0001,0.001]:
            for dropout in [1.0,0.9,0.8]:
                for activation in ["tanh","relu"]:
                    for train_size in [500000]:
                        config.lr = lr
                        config.l2 = l2
                        config.dropout = dropout
                        config.activation= activation
                        with tf.Session() as sess:
                            tf.global_variables_initializer().run()
                            hypaccs= []
                            results = []
                            best_preds = []
                            best_acc = 0
                            count = 0
                            for ep in range(int((config.num_epoch*500000)/train_size)):
                                print("\n>> Beginning epoch {}/{} <<".format(ep, config.num_epoch))
                                ### Call training and validation routine
                                ### This part runs one epoch of trainin#g and one epoch of validation
                                ### Outcomes: preds, labels, constr, loss
                                train_data = model_util.get_feed(os.path.join(config.data_path, experiment + ".train"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len,num_iter=int(train_size/config.batch_size), shuffle = True)
                                for prem, prem_len, hyp, hyp_len, label in train_data:
                                    pred, _ = m.optimize(sess, prem, prem_len, hyp, hyp_len, config.dropout, label, config.lr, config.l2)
                                    count += 1
                                    if count*config.batch_size % 100000 < config.batch_size:
                                        val_data = model_util.get_feed(os.path.join(config.data_path,  experiment +".val"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len)
                                        preds_val, labels_val, _= m.run_test_epoch(sess, val_data)
                                        train_data2 = model_util.get_feed(os.path.join(config.data_path, experiment + ".train"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len, num_iter=int(min(train_size,10000)/config.batch_size), shuffle = False)
                                        preds_train2, labels_train2, _= m.run_test_epoch(sess, train_data2)
                                        results.append((ratio(preds_train2, labels_train2), ratio(preds_val, labels_val), confuse(preds_train2, labels_train2), confuse(preds_val, labels_val)))
                                        if ratio(preds_val, labels_val) > best_acc:
                                            best_acc = ratio(preds_val, labels_val)
                                            best_preds = preds_val
                                        hypaccs.append(ratio(preds_val,labels_val))
                            hyp_results[(str(lr),str(l2),str(activation), str(dropout))] = max(hypaccs)
                    for k in hyp_results:
                        print(k, ":    ",hyp_results[k])
                                #test_data = model_util.get_feed(os.path.join(config.data_path, experiment + ".test"), config.batch_size, word_to_id, config.max_prem_len, config.max_hyp_len, num_iter=2)
                                #preds_test, labels_test, loss_test = m.run_test_epoch(sess, test_data)
                            #            with open("compvary" + experiment + str(lr) + str(l2) + str(batch_size), "w") as f:
                            #                f.write(json.dumps((results, [int(pred) for pred in best_preds])))
