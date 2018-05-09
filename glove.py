import random, collections
import numpy as np

def _get_glove_vocab(glovepath, vocab_limit=None):
    word_to_id = collections.defaultdict(lambda: 0)
    word_to_id['<unk>'] = 0
    word_to_id['<blank>'] = 1
    word_to_id['<eos>'] = 2
    with open(glovepath, 'r', encoding = "utf8") as f:
        i = 3
        for line in f:
            word_to_id[line[:-1]] = i
            i += 1
            if i % 20000 == 0: print("{} words read".format(i))
            if vocab_limit is not None and i >= vocab_limit: break
        return word_to_id

def _get_glove_vec(dim=300, vocab_limit=None):
    mat = []
    with open(r"glove\glove.6B.{}d.txt".format(dim), "r", encoding="utf8") as f:
        i = 0
        for line in f:
            mat.append(list(map(float, line[:-1].split(' ')[1:])))
            i += 1
            if i % 20000 == 0: print("{} words read".format(i))
            if vocab_limit is not None and i + 3 >= vocab_limit: break
    unk_vec = random.choice(mat) # need deepcopy if not converted to numpy
    blank_vec = random.choice(mat)
    eos_vec = random.choice(mat)
    return np.array([unk_vec] + [blank_vec] + [eos_vec] + mat, dtype=np.float32)
