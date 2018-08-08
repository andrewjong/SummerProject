import generate_data as gd
import numpy as np
import random
import json
import itertools
import math
import data_util as du

def sentence_to_id(sentence, word_to_id, max_len):
    ret = [word_to_id[word.lower()] for word in sentence.split()[:max_len]]
    return ret + [1] * (max_len -len(ret))

def label_to_num(l):
    d = {'entails':0, 'contradicts':1, 'permits':2}
    return d[l]

def get_feed(path, batch_size, word_to_id, max_premise_length, max_hypothesis_length, num_iter = None, shuffle = False):
    data,_,_ = gd.process_data(1.0)
    premises = []
    premise_lengths = []
    hypotheses = []
    hypothesis_lengths = []
    labels = []
    with open(path,'r') as f:
        lines = f.readlines()
        if shuffle:
            random.shuffle(lines)
        for line in lines:
            example = json.loads(line)
            premises.append(sentence_to_id(du.parse_sentence(data,example["sentence1"])[0].emptystring, word_to_id, max_premise_length))
            premise_lengths.append(len(du.parse_sentence(data,example["sentence1"])[0].emptystring.split()))
            hypotheses.append(sentence_to_id(du.parse_sentence(data,example["sentence2"])[0].emptystring, word_to_id, max_hypothesis_length))
            hypothesis_lengths.append(len(du.parse_sentence(data,example["sentence2"])[0].emptystring.split()))
            labels.append(label_to_num(example["gold_label"]))
            if num_iter is not None and len(labels) > num_iter*batch_size:
                break
    if num_iter is None:
        num_iter = int(math.ceil(len(labels)/ batch_size))
    for i in range(num_iter):
        yield (np.array(premises[i * batch_size: (i+1) * batch_size]),
               np.array(premise_lengths[i * batch_size: (i+1) * batch_size]),
               np.array(hypotheses[i * batch_size: (i+1) * batch_size]),
               np.array(hypothesis_lengths[i * batch_size: (i+1) * batch_size]),
               np.array(labels[i * batch_size: (i+1) * batch_size]))

def get_vocab():
    data, _, _ = gd.process_data(1.0)
    vocab = ["does", "not", "any", "or", "and", "if", "then"]
    for k in data:
        for word in data[k]:
            if type(word) == list:
                vocab += [w.lower() for w in word]
            else:
                vocab.append(word.lower())
    return vocab

def get_word_to_id(glovepath, vocab):
    word_to_id = dict()
    word_to_id["emptystring"] = 0
    word_to_id["notevery"] = 1
    for i, word in enumerate(vocab):
        word_to_id[word] = i + 2
    return word_to_id

def get_id_to_word(glovepath, vocab):
    d = _get_word_to_id(glovepath, vocab)
    result = {}
    for word in d:
        result[d[word]] = word
    return result

def get_glove_vec(glovepath, vocab):
    all_words = dict()
    with open(glovepath, "r", encoding="utf8") as f:
        for line in f:
            word = str(line.split()[0])
            all_words[word] = [ float(number) for number in line.split()[1:]]
    mat = []
    mat.append([random.uniform(-1,1) for _ in range(300)])
    mat.append([random.uniform(-1,1) for _ in range(300)])
    for word in vocab:
        if word in all_words:
            mat.append(all_words[word])
        else:
            mat.append([random.uniform(-1,1) for _ in range(300)])
    return np.array(mat, dtype=np.float32)
