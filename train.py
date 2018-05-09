from __future__ import absolute_import, division, print_function
import json
import scipy
import sys, os, time, random, collections, itertools
import tensorflow as tf
import util
import math
import numpy as np
from model import PIModel
from configs.standard_conf import config
from new_generate_data import sentence


def main(args):
    #exec("from configs.{} import config".format(sys.argv[1]))

    ## get embedding
    word_to_id = util._get_word_to_id("glove/glove.6B.list", vocab_limit=config.vocab_limit)
    pretrained_embeddings = util._get_glove_vec("glove/glove.6B.300d.txt", vocab_limit=config.vocab_limit)

    ## Initialize model and tf session
    model_type = args[1] if len(args)>1 else 'seq2seq'

    m = PIModel(config, pretrained_embeddings, model_type)
    labels = ['entails','contradicts','permits']
    cat_names = ['{}=>{}'.format(x,y) for x,y in itertools.product(labels,labels)]
    logs = []
    trains = []
    joints = ["simplejoint", "simpledisjoint"]
    types = ["training_data"]
    search = False
    for att in ["seq2seq"]:
        if att == "attention":
            config.attention = True
        else:
            config.attention = False
        model_type = att
        for t in types:
            config.data_path = t
            for joint in joints:
                for time in range(5):
                    info =[] 
                    best = (0, 0)
                    bestd = dict()
                    param = 0
                    lol = []
                    config.l2 = 0
                    config.dropout = 1
                    config.lr = 0.001
                    filename = att + "trainsize" + "_" + str(time) + "_" + joint + "_" + t[:4]
                    with tf.Session() as sess:
                        tf.global_variables_initializer().run()
                        ## Iterate
                        print("\n-----\n")
                        print("Learning Rate:", config.lr)
                        print("Dropout:", config.dropout)
                        print("L2:", config.l2)
                        print(t, joint)
                        for ep in range(config.num_epoch):
                            print("\n>> Beginning epoch {}/{} <<".format(ep, config.num_epoch))
                            content = dict()

                            ### Call training and validation routine
                            ### This part runs one epoch of trainin#g and one epoch of validation
                            ### Outcomes: preds, labels, constr, loss
                            train_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='train', prefix='pi', shuffle=False, joint=joint)
                            preds_t, labels_t, loss_t = m.run_train_epoch(sess, train_data, [config.lr], config.dropout, config.l2)
                            val_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='val', prefix='pi', shuffle=False, joint=joint)
                            preds_v, labels_v, loss_v = m.run_val_epoch(sess, val_data)
                            test_data = util.get_feed(config.data_path, batch_size=config.batch_size, max_prem_len=config.max_prem_len, max_hyp_len=config.max_hyp_len, word_to_id=word_to_id, mode='test', prefix='pi', shuffle=False, joint=joint)
                            preds_v2, labels_v2, loss_v2 = m.run_val_epoch(sess, test_data)

                            ### This part evaluates the model performance: predictions (preds) against labels (labels)
                            ###     metric: f1_macro, f1_micro, accuracy, confusion_matrix
                            ###     by_construct: Want results separately for each implicative construct?
                            ###     constr: (used when by_consruct is true) the source constructs for each prediction/label pairs
                            s0 = util.get_evaluation(preds_t, labels_t, metric='f1_macro')
                            s1 = util.get_evaluation(preds_t, labels_t, metric='f1_micro')
                            s2 = util.get_evaluation(preds_v, labels_v, metric='f1_macro')
                            s3 = util.get_evaluation(preds_v, labels_v, metric='f1_micro')

                            s_cat = util.get_evaluation(preds_v, labels_v, metric='confusion_matrix')

                            #for constr in s_cat:
                            #    logs[constr].append(s_cat[constr])  # write confusion matrix to log
                            ### Print the performance
                            val_constr = []
                            nest_len = []
                            train_constr = []
                            #val_neg = []
                            #train_neg = []
                            train_results = []
                            test_results = []
                            print("Train loss = {} :: Val loss = {}".format(loss_t, loss_v))
                            print("Train F1 macro/micro = {}/{} :: Val F1 macro/micro = {}/{}".format(s0, s1, s2, s3))
                            constr = dict()
                            right = 0
                            total = 0
                            c = 0
                            ct =0
                            e = 0
                            et = 0
                            p = 0
                            pt = 0
                            train_conf = [[0,0,0],[0,0,0],[0,0,0]]
                            for i in range(len(preds_t)):
                                x = preds_t[i]
                                y = labels_t[i]
                                train_conf[x][y] += 1
                                if train_constr[i] not in constr:
                                    constr[train_constr[i]] = [0,0]
                                if "train_conf_" + train_constr[i] not in content:
                                    content["train_conf_" + train_constr[i]] = [[0,0,0],[0,0,0],[0,0,0]]
                                content["train_conf_" + train_constr[i]][x][y] +=1
                                #if train_neg[i] != "ignore":
                                #    if  train_neg[i]+"train_conf_" + train_constr[i] not in content:
                                #        content[train_neg[i]+"train_conf_" + train_constr[i]] = [[0,0,0],[0,0,0],[0,0,0]]
                                #    content[train_neg[i]+"train_conf_" + train_constr[i]][x][y] +=1
                                train_results.append((x,y))
                                if y == 0:
                                    et += 1
                                if y == 1:
                                    ct += 1
                                if y == 2:
                                    pt += 1
                                if x == y:
                                    constr[train_constr[i]][0] +=1
                                    right += 1
                                    if y == 0:
                                        e += 1
                                    if y == 1:
                                        c += 1
                                    if y == 2:
                                        p += 1
                                else:
                                    constr[train_constr[i]][1] +=1
                                total += 1
                            train_acc = right/total
                            print("\nTraining accuracy:", train_acc)
                            print("Entails:", e/et)
                            print("Contradicts:", c/ct)
                            print("Permits:", p/pt)
                            constr2 = dict()
                            right = 0
                            total = 0
                            c = 0
                            ct =0
                            e = 0
                            et = 0
                            p = 0
                            pt = 0
                            val_conf = [[0,0,0],[0,0,0],[0,0,0]]
                            for i in range(len(preds_v)):
                                x = preds_v[i]
                                y = labels_v[i]
                                val_conf[x][y] += 1
                                if val_constr[i] not in constr2:
                                    constr2[val_constr[i]] = [0,0]
                                if "val_conf_" + val_constr[i] not in content:
                                    content["val_conf_" + val_constr[i]] = [[0,0,0],[0,0,0],[0,0,0]]
                                content["val_conf_" + val_constr[i]][x][y] += 1
                                #if val_neg[i] != "ignore":
                                #    if val_neg[i] + "val_conf_" + val_constr[i] not in content:
                                #        content[val_neg[i]+"val_conf_" + val_constr[i]] = [[0,0,0],[0,0,0],[0,0,0]]
                                #    content[val_neg[i]+ "val_conf_" + val_constr[i]][x][y] += 1
                                if y == 0:
                                    et += 1
                                if y == 1:
                                    ct += 1
                                if y == 2:
                                    pt += 1
                                if x == y:
                                    constr2[val_constr[i]][0] +=1
                                    right += 1
                                    if y == 0:
                                        e += 1
                                    if y == 1:
                                        c += 1
                                    if y == 2:
                                        p += 1
                                else:
                                    constr2[val_constr[i]][1] +=1
                                total += 1
                            print("\nValidation Accuracy:", right/total)
                            e_acc = 0
                            c_acc = 0
                            p_acc = 0
                            if right/total > best[0]:
                                best = (s0, s1, s2,s3)
                                bestd = (constr, constr2)
                                param = (config.l2, config.dropout, config.lr, ep)
                                e_acc = e/et
                                c_acc = c/ct
                                p_acc = p/pt
                                lol = lenresult
                                temp1 = []
                                temp2 = []
                                for i in preds_v:
                                    temp1.append(int(i))
                                for i in labels_v:
                                    temp2.append(int(i))
                            print("Entails Accuracy:", e/et)
                            print("Contradicts Accuracy:", c/ct)
                            print("Permits Accuracy:", p/pt)
                            mat = s_cat / np.sum(s_cat, axis=1, keepdims=True)
                            if False:
                                time_train_done = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                                dest_dir = os.path.join("results", "{}-{}".format(sys.argv[1], time_train_done))
                                os.makedirs(dest_dir)
                                print(tf.train.Saver().save(sess, os.path.join(dest_dir, 'model' + str(ep))))
                                with open(os.path.join(dest_dir,'log'), 'w') as f:
                                    for cat_name, scores in zip(cat_names, zip(*logs)):
                                        f.write(cat_name + " " + " ".join(map(str, scores)) + "\n")
                                    for train in trains:
                                        f.write(train + "\n")
                content["train_macro"] = best[0]
                content["train_micro"] = best[1]
                content["val_macro"] = best[2]
                content["val_micro"] = best[3]

                for k in bestd[1]:
                    content["test_" +  k] = float(bestd[1][k][0])/float(bestd[1][k][0] + bestd[1][k][1])
                for k in bestd[0]:
                    content["train_" +  k] = float(bestd[0][k][0])/float(bestd[0][k][0] + bestd[0][k][1])
                content["param"] = param
                with open(filename, "w") as f:
                    f.writelines([json.dumps(content)])



if __name__ == '__main__':
    tf.app.run()
