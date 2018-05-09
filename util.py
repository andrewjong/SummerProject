from __future__ import print_function, division, absolute_import
import sys, os, random, itertools, collections, math
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle

def retrieve_from(path, joint, prefix="pi"):
    '''
    Read the files in the path and return a dictionary
    from splitnames (str) -> list of inputs in raw string
    '''
    portion = ['train','val','test', "nest" ]
    part = ['sentence1','gold_label','sentence2']
    potential_names = [(x,y) for x,y in itertools.product(part, portion)]
    file_paths = [os.path.join(path, joint + "." +  n[1] ) for n in potential_names]
    ret_dict = {}
    for fname, n in zip(file_paths, potential_names):
        try:
            with open(fname,'rb') as f:
                lines = pickle.loads(pickle.load(f))
                temp = []
                for line in lines:
                    temp.append(line[n[0]])
                ret_dict[n] = temp
        except IOError:
            pass
    return ret_dict

def get_pretrain_data():
    data = dict() 
    data["prem"] = np.random.randint(0, high=65535, size = (256, 25))
    data["prem_len"] = np.random.randint(1, high=25, size = (256,))
    data["hyp"] = data["prem"]#np.random.randint(0, high=65535, size = (25600, 25))
    data["hyp_len"] = data["prem_len"]#np.random.randint(0, high=25, size = (25600,))
    data["label"] = np.zeros((256,))
    #indices = np.random.choice(256, size=(128,), replace = False)
    #data["hyp"][indices] = data["prem"][indices]
    #data["hyp_len"][indices] = data["prem_len"][indices]
    data["constr"] = ["none"] * 256
    return data
        

def _sent_to_id(sent, word_to_id, max_len):
    ret = list(map(lambda x: word_to_id[x], *zip(*itertools.zip_longest(sent.split(' ')[:max_len]))))
    return ret + [1] * (max_len -len(ret)), min(len(ret), max_len)

def _id_to_sent(ids, id_to_word):
    return ' '.join(list(map(lambda x: id_to_word[x], *zip(*itertools.zip_longest(ids)))))

def _label_to_num(l):
    d = {'entails':0, 'contradicts':1, 'permits':2}
    return d[l]

def _num_to_label(l):
    d = ['entails','contradicts','permits']
    return d[l]

def get_masked_data(path, **kwargs):
    max_prem_len = kwargs['max_prem_len']
    max_hyp_len = kwargs['max_hyp_len']
    word_to_id = kwargs['word_to_id']
    mode = kwargs['mode']
    prefix = kwargs['prefix']
    dat = retrieve_from(path, kwargs["joint"], prefix=prefix)
    prem, prem_len = zip(*list(map(lambda x: _sent_to_id(x, word_to_id, max_prem_len), *zip(*itertools.zip_longest(dat[("sentence1", mode)])))))
    hyp, hyp_len = zip(*list(map(lambda x: _sent_to_id(x, word_to_id, max_hyp_len), *zip(*itertools.zip_longest(dat[("sentence2", mode)])))))
    return {
        "prem": prem,
        "prem_len": prem_len,
        "hyp": hyp,
        "hyp_len": hyp_len,
        "label": list(map(_label_to_num, *zip(*itertools.zip_longest(dat[("gold_label", mode)]))))
    }

def get_feed(path, batch_size, **kwargs):
    dat = get_masked_data(path, **kwargs)
    if kwargs['shuffle']:
        ind = list(range(len(dat["label"])))
        random.shuffle(ind)
        dat = {l:[dat[l][i] for i in ind] for l in dat}
    num_iter = int(math.ceil(len(dat["label"]) / batch_size))
    for i in range(num_iter):
        yield (np.array(dat['prem'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['prem_len'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['hyp'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['hyp_len'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['label'][i * batch_size: (i+1) * batch_size]))


def get_feed2():
    batch_size = 256
    dat = get_pretrain_data()
    num_iter = int(math.ceil(len(dat["label"]) / 256))
    for i in range(num_iter):
        yield (np.array(dat['prem'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['prem_len'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['hyp'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['hyp_len'][i * batch_size: (i+1) * batch_size]),
               np.array(dat['label'][i * batch_size: (i+1) * batch_size]),
               dat['constr'][i * batch_size: (i+1) * batch_size])

def _get_word_to_id(glovepath, vocab_limit=None):
    word_to_id = collections.defaultdict(lambda: 0)
    word_to_id['<blank>'] = 1
    with open(glovepath, 'r') as f:
        i = 2
        for line in f:
            word_to_id[line[:-1]] = i
            i += 1
            #if i % 20000 == 0: print("{} words read".format(i))
            if vocab_limit is not None and i >= vocab_limit: break
        return word_to_id

def _get_id_to_word(glovepath, vocab_limit=None):
    d = _get_word_to_id(glovepath, vocab_limit)
    result = {}
    for word in d:
        result[d[word]] = word
    return result

def _get_glove_vec(glovepath, vocab_limit=None):
    mat = []
    with open(glovepath, "r") as f:
        i = 0
        for line in f:
            mat.append(list(map(float, *zip(*itertools.zip_longest(line[:-1].split(' ')[1:])))))
            i += 1
            #if i % 20000 == 0: print("{} words read".format(i))
            if vocab_limit is not None and i + 2 >= vocab_limit:
                break
    unk_vec = random.choice(mat) # need deepcopy if not converted to numpy
    blank_vec = random.choice(mat)
    return np.array([unk_vec] + [blank_vec] + mat, dtype=np.float32)

def get_evaluation(preds, labels, **kwargs):
    metric_map = {
        "confusion_matrix" : lambda x,y: confusion_matrix(x,y),
        "f1_macro" : lambda x,y: f1_score(x,y,average="macro"),
        "f1_micro" : lambda x,y: f1_score(x,y,average="micro"),
        "accuracy" : lambda x,y: accuracy_score(x,y)
    }
    return metric_map[kwargs["metric"]](labels,preds)

if __name__ == '__main__':
    word_to_id = _get_word_to_id("glove/glove.6B.list", vocab_limit=65536)
    g = get_feed("disjoint", 2, max_prem_len=24, max_hyp_len=24, word_to_id=word_to_id, mode='train', prefix='pi', shuffle=True)
    print(g.next())
