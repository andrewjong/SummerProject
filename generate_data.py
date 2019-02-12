from data_util import sentence
from data_util import parse_simple_sentence
import natural_logic_model as nlm
import os
import random
import json
import numpy as np
from functools import reduce

def process_data(train_ratio):
    #split the different parts of speech into train, validation, and test
    #determiners are not split
    train = dict()
    val = dict()
    test = dict()
    categories = ["agents", "transitive_verbs", "things", "determiners", "adverbs", "subject_adjectives","object_adjectives"]
    for c in categories:
        with open(os.path.join("data", c + ".txt"),"r") as f:
            stuff = f.readlines()
            if c != "transitive_verbs":
                stuff = [_.strip() for _ in stuff]
            else:
                stuff = [_.strip().split() for _ in stuff]
        random.shuffle(stuff)
        if c != "determiners":
            train[c] = stuff[:int(len(stuff)*train_ratio)]
            val[c] = stuff[int(len(stuff)*train_ratio):int(len(stuff)*(train_ratio+(1-train_ratio)*0.5))]
            test[c] = stuff[int(len(stuff)*(train_ratio+(1-train_ratio)*0.5)):]
        else:
            train[c] = stuff
            val[c] = stuff
            test[c] = stuff
    return train, val, test

def save_data(examples, name):
    #saves data in the SNLI format
    data = []
    for example in examples:
        example_dict = dict()
        example_dict["sentence1"] = example[0]
        example_dict["sentence2"] = example[2]
        example_dict["gold_label"] = example[1]
        data.append(json.dumps(example_dict))
    with open(name, 'w') as f:
        for datum in data:
            f.write(datum + "\n")

def restricted(restrictions, enc):
    #This function determines whether an encoding of an NLI input
    #is restricted according to restrictions
    if restrictions is None:
        return False
    if enc in restrictions:
        return False
    return True

def split_dict(filename, restrictions):
    #This function takes in a dictionary generated by build_simple_file or
    # build_boolean_file and divides the encoded NLI input keys by the label
    #they are mapped to
    with open(filename, 'r') as f:
        solutions= json.loads(f.read())
    e = dict()
    c = dict()
    p = dict()
    for i in solutions:
        if restricted(restrictions,i):
            continue
        if solutions[i] == "entails":
            e[i] = solutions[i]
        if solutions[i] == "contradicts":
            c[i] = solutions[i]
        if solutions[i] == "permits":
            p[i] = solutions[i]
    return e, c, p

def compute_relation(lexicon, relation_index):
    #This function takes in a lexicon list of words and a relation_index
    #and outputs the same random word twice if relation_index is 0,
    #the empty string and a random word if relation_index is 1,
    #the a random word and the empty string if relation_index is 2,
    #and two different random words if relation index is 3
    if relation_index == 0:
        premise_word = random.choice(lexicon + [""])
        hypothesis_word = premise_word
    if relation_index == 1:
        premise_word = ""
        hypothesis_word = random.choice(lexicon)
    if relation_index == 2:
        premise_word = random.choice(lexicon)
        hypothesis_word = ""
    if relation_index == 3:
        premise_word = random.choice(lexicon)
        hypothesis_word = select_new(lexicon, premise_word)
    return premise_word, hypothesis_word

def select_new(lexicon, old):
    #returns an element of lexicon that is not old without changing lexicon
    index = lexicon.index(old)
    lexicon.remove(old)
    new = random.choice(lexicon)
    lexicon.insert(index, old)
    return new

def encoding_to_independent_example(data, encoding, premise, hypothesis):
    new_premise, new_hypothesis = encoding_to_example(data,encoding)
    while nlm.compute_simple_relation(premise, new_premise) != "independence" or nlm.compute_simple_relation(premise, new_hypothesis) != "independence" or nlm.compute_simple_relation(hypothesis, new_hypothesis) != "independence" or nlm.compute_simple_relation(hypothesis, new_premise) != "independence":
        new_premise, new_hypothesis = encoding_to_example(data,encoding)
    return new_premise, new_hypothesis


def encoding_to_example(data, encoding):
    #takes in an encoding produced by build_simple_file
    #and outputs two sentence objects corresponding to the encoding
    dets = ["every", "not every", "some", "no"]
    psubject_noun = random.choice(data["agents"])
    pverb = random.choice(data["transitive_verbs"])
    pobject_noun = random.choice(data["things"])
    hsubject_noun = psubject_noun
    hverb = pverb
    hobject_noun = pobject_noun
    if encoding[-3] == 0:
        hsubject_noun = select_new(data["agents"], psubject_noun)
    if encoding[-2] == 0:
        hverb = select_new(data["transitive_verbs"], pverb)
    if encoding[-1] == 0:
        hobject_noun = select_new(data["things"], pobject_noun)
    padverb, hadverb  = compute_relation(data["adverbs"], encoding[-4])
    pobject_adjective, hobject_adjective = compute_relation(data["object_adjectives"], encoding[-5])
    psubject_adjective, hsubject_adjective = compute_relation(data["subject_adjectives"], encoding[-6])
    return sentence(psubject_noun, pverb, pobject_noun, encoding[0], padverb, psubject_adjective, pobject_adjective, dets[encoding[1]],dets[encoding[2]]), sentence(hsubject_noun, hverb, hobject_noun, encoding[3], hadverb, hsubject_adjective, hobject_adjective, dets[encoding[4]],dets[encoding[5]])

def example_to_encoding(premise, hypothesis):
    encoding = []
    dets = ["every", "not every", "some", "no"]
    if premise.negation == "does not":
        encoding.append(1)
    else:
        encoding.append(0)
    encoding += [dets.index(premise.subject_determiner),dets.index(premise.object_determiner)]
    if hypothesis.negation == "does not":
        encoding.append(1)
    else:
        encoding.append(0)
    encoding += [dets.index(hypothesis.subject_determiner),dets.index(hypothesis.object_determiner)]
    if premise.subject_adjective == hypothesis.subject_adjective:
        encoding.append(0)
    elif premise.subject_adjective == "":
        encoding.append(1)
    elif hypothesis.subject_adjective == "":
        encoding.append(2)
    else:
        encoding.append(3)
    if premise.object_adjective == hypothesis.object_adjective:
        encoding.append(0)
    elif premise.object_adjective == "":
        encoding.append(1)
    elif hypothesis.object_adjective == "":
        encoding.append(2)
    else:
        encoding.append(3)
    if premise.adverb == hypothesis.adverb:
        encoding.append(0)
    elif premise.adverb == "":
        encoding.append(1)
    elif hypothesis.adverb == "":
        encoding.append(2)
    else:
        encoding.append(3)
    if premise.subject_noun == hypothesis.subject_noun:
        encoding.append(1)
    else:
        encoding.append(0)
    if premise.verb == hypothesis.verb:
        encoding.append(1)
    else:
        encoding.append(0)
    if premise.object_noun == hypothesis.object_noun:
        encoding.append(1)
    else:
        encoding.append(0)
    return encoding

def gcd(a, b):
    if not b:
        return a
    else:
        return gcd(b, a % b)

def gcd_n(numbers):
    return reduce(lambda x, y: gcd(x, y), numbers)

def get_boolean_encoding_counts(bool_keys, keys_and_counts, level):
    hard_bools = [(0, 0, 0, 6), (1, 1, 0, 6), (1, 2, 0, 6), (2, 1, 0, 6), (2, 2, 0, 6), (0, 0, 1, 6), (1, 1, 1, 6), (1, 2, 1, 6), (1, 0, 2, 6), (1, 1, 2, 6), (1, 2, 2, 6), (2, 1, 2, 6), (2, 2, 2, 6), (0, 1, 4, 6), (0, 2, 4, 6), (1, 0, 4, 6), (2, 0, 4, 6), (1, 0, 5, 6), (1, 1, 5, 6), (1, 2, 5, 6), (2, 0, 5, 6), (0, 1, 3, 6), (0, 2, 3, 6), (1, 0, 3, 6), (0, 0, 6, 0), (0, 2, 6, 0), (1, 1, 6, 0), (2, 2, 6, 0), (0, 0, 6, 1), (0, 2, 6, 1), (1, 1, 6, 1), (2, 2, 6, 1), (1, 0, 6, 2), (1, 1, 6, 2), (1, 2, 6, 2), (0, 1, 6, 4), (1, 0, 6, 4), (1, 2, 6, 4), (2, 1, 6, 4), (1, 0, 6, 5), (1, 1, 6, 5), (1, 2, 6, 5), (0, 1, 6, 3), (1, 0, 6, 3), (1, 2, 6, 3), (2, 1, 6, 3), (1, 0, 6, 6), (1, 0, 6, 6), (1, 1, 6, 6), (1, 1, 6, 6), (1, 2, 6, 6), (1, 2, 6, 6)]
    counts = []
    balance_dict = dict()
    total = 0
    for i in range(3):
        for j in range(3):
            balance_dict[(i,j)] = 0
    for encoding in bool_keys:
        balance_dict[tuple(json.loads(encoding)[:2])] += 1
    init = True
    lcm = 0
    for k in balance_dict:
        if balance_dict[k] == 0:
            continue
        elif init:
            lcm = balance_dict[k]
            init = False
        else:
            lcm = lcm*balance_dict[k]/gcd(lcm, balance_dict[k])
    for encoding in bool_keys:
        encoding = json.loads(encoding)
        if level == "level 0":
            for i in range(7):
                if encoding[2] == i:
                    first_simple = sum(keys_and_counts[1][i])
                if encoding[3] == i:
                    second_simple = sum(keys_and_counts[1][i])
            counts.append(first_simple * second_simple)
        if level == "level 2":
            if tuple(encoding) in hard_bools:
                counts.append(20)
            else:
                counts.append(1)
            #counts.append(lcm/balance_dict[tuple(encoding[:2])])

    full_gcd = gcd_n(counts)
    counts = [count/full_gcd for count in counts]
    return counts


def generate_balanced_boolean_data(bool_keys, label, keys_and_counts, sampling,size, data):
    #using encoded compound examples in boolkeys, and the encoded simple
    #examples in ekeys, ckeys, and pkeys, this function outputs a list of length
    #size with compound sentence examples
    result = []
    if sampling == "level 0" or sampling == "level 2":
        bool_counts = get_boolean_encoding_counts(bool_keys, keys_and_counts, sampling)
    elif sampling == "level 1":
        bool_counts = [1] * len(bool_keys)
    for i in range(size):
        encoding = json.loads(weighted_selection(bool_keys, bool_counts))
        for i in range(7):
            if encoding[2] == i:
                simple1_encoding = json.loads(weighted_selection(keys_and_counts[0][i], keys_and_counts[1][i]))
                premise1, hypothesis1 = encoding_to_example(data, simple1_encoding)
        for i in range(7):
            if encoding[3] == i:
                simple2_encoding = json.loads(weighted_selection(keys_and_counts[0][i], keys_and_counts[1][i]))
                premise2, hypothesis2 = encoding_to_independent_example(data, simple2_encoding, premise1, hypothesis1)
        conjunctions = ["or", "and", "then"]
        premise_conjunction = conjunctions[encoding[0]]
        hypothesis_conjunction = conjunctions[encoding[1]]
        premise_compound = premise1.string + " " + premise_conjunction + " " + premise2.string
        hypothesis_compound = hypothesis1.string+ " " + hypothesis_conjunction+ " " + hypothesis2.string
        if premise_conjunction == "then":
            premise_compound = "if " + premise_compound
        if hypothesis_conjunction == "then":
            hypothesis_compound = "if " + hypothesis_compound
        result.append((premise_compound, label, hypothesis_compound))
    return result

def sevenclass_simple_encodings(data,simple_ratio, ekeys, ckeys, pkeys, ecounts, ccounts, pcounts):
    #This function trims simple nli encodings for use in
    #generating compound sentences see paper for why this is necessary
    new_eqkeys = []
    new_ekeys = []
    new_rekeys = []
    new_akeys = []
    new_ckeys = []
    new_cokeys = []
    new_pkeys = []
    new_eqcounts= []
    new_ecounts= []
    new_recounts= []
    new_acounts = []
    new_ccounts = []
    new_cocounts= []
    new_pcounts = []
    for encoding, count in zip(ekeys, ecounts):
        premise, hypothesis = encoding_to_example(data,json.loads(encoding))
        if nlm.compute_simple_relation(premise, hypothesis) == "entails" and random.uniform(0,1) < simple_ratio:
            new_ekeys.append(encoding)
            new_ecounts.append(int(count))
        else:
            new_eqkeys.append(encoding)
            new_eqcounts.append(int(count))
    for encoding, count in zip(ckeys, ccounts):
        premise, hypothesis = encoding_to_example(data,json.loads(encoding))
        if nlm.compute_simple_relation(premise, hypothesis) == "alternation" and random.uniform(0,1) < simple_ratio:
            new_akeys.append(encoding)
            new_acounts.append(int(count))
        else:
            new_ckeys.append(encoding)
            new_ccounts.append(int(count))
    for encoding, count in zip(pkeys, pcounts):
        premise, hypothesis = encoding_to_example(data,json.loads(encoding))
        if nlm.compute_simple_relation(premise, hypothesis) == "independence" and random.uniform(0,1) < simple_ratio:
            new_pkeys.append(encoding)
            new_pcounts.append(int(count))
        elif nlm.compute_simple_relation(premise, hypothesis) == "cover" and random.uniform(0,1) < simple_ratio:
            new_cokeys.append(encoding)
            new_cocounts.append(int(count))
        else:
            new_rekeys.append(encoding)
            new_recounts.append(int(count))
    return ((new_eqkeys, new_ekeys, new_rekeys, new_akeys, new_ckeys, new_cokeys, new_pkeys), (new_eqcounts,new_ecounts,new_recounts,new_acounts, new_ccounts, new_cocounts, new_pcounts))


def level0_example_count(data, encoding):
    count = 1
    noun_object_size = len(data["things"])
    verb_size = len(data["transitive_verbs"])
    noun_subject_size = len(data["agents"])
    subject_adjective_size = len(data["subject_adjectives"])
    object_adjective_size = len(data["object_adjectives"])
    adverb_size = len(data["adverbs"])
    if encoding[-1] == 1:
        count *= noun_object_size
    else:
        count *= noun_object_size * noun_object_size - noun_object_size
    if encoding[-2] == 1:
        count *= verb_size
    else:
        count *= verb_size * verb_size - verb_size
    if encoding[-3] == 1:
        count *= noun_subject_size
    else:
        count *= noun_subject_size * noun_subject_size - noun_subject_size
    if encoding[-4] == 0:
        count *= adverb_size + 1
    elif encoding[-4] == 1 or encoding[-4] == 2:
        count *= adverb_size
    else:
        count *= (adverb_size + 1)^2 - 3* adverb_size - 1
    if encoding[-5] == 0:
        count *= object_adjective_size + 1
    elif encoding[-5] == 1 or encoding[-5] == 2:
        count *= object_adjective_size
    else:
        count *= (object_adjective_size + 1)^2 - 3* object_adjective_size - 1
    if encoding[-6] == 0:
        count *= subject_adjective_size + 1
    elif encoding[-6] == 1 or encoding[-6] == 2:
        count *= subject_adjective_size
    else:
        count *= (subject_adjective_size + 1)^2 - 3* subject_adjective_size - 1
    return count

def level2_example_counts(data, ekeys,ckeys,pkeys):
    weights = np.zeros((4,4,4))
    for encoding in ekeys:
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        weights[x,y,z] +=1
    for encoding in ckeys:
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        weights[x,y,z] +=1
    for encoding in pkeys:
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        weights[x,y,z] +=1
    print("hey")
    ecounts = []
    ccounts = []
    pcounts = []
    for encoding in ekeys:
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        ecounts.append(np.sum(weights)/weights[x,y,z])
    for encoding in ckeys:
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        ccounts.append(np.sum(weights)/weights[x,y,z])
    for encoding in pkeys:
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        pcounts.append(np.sum(weights)/weights[x,y,z])
    weights = np.zeros((4,4,4))
    for i, encoding in enumerate(ekeys):
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        weights[x,y,z]+=ecounts[i]
    for i, encoding in enumerate(ckeys):
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        weights[x,y,z]+=ccounts[i]
    for i,encoding in enumerate(pkeys):
        encoding = json.loads(encoding)
        if encoding[-1] == 1:
            x = encoding[-5]
        else:
            x = 3
        if encoding[-2] == 1:
            y = encoding[-4]
        else:
            y = 3
        if encoding[-3] == 1:
            z = encoding[-6]
        else:
            z = 3
        weights[x,y,z]+=pcounts[i]
    print("hey")
    #for i in range(4):
        #for j in range(4):
            #for k in range(4):
                #print(weights[x,y,z])
    return ecounts, ccounts, pcounts

def get_simple_encoding_counts(data, level, ekeys, ckeys, pkeys):
    ecounts = []
    ccounts = []
    pcounts = []
    if level == "level 0":
        ecounts, ccounts, pcounts = level0_example_count(data, ekeys,ckeys,pkeys)
    else:
        ecounts, ccounts, pcounts= level2_example_counts(data, ekeys,ckeys,pkeys)
    gcd = gcd_n(ecounts)
    ecounts = [ecount/gcd for ecount in ecounts]
    gcd = gcd_n(ccounts)
    ccounts = [ccount/gcd for ccount in ccounts]
    gcd = gcd_n(pcounts)
    pcounts = [pcount/gcd for pcount in pcounts]
    return ecounts, ccounts, pcounts

def weighted_selection(keys, counts):
    total = sum(counts)
    x = random.randint(1,total)
    for key, count in zip(keys, counts):
        x -= count
        if x <= 0:
            return key

def generate_balanced_data(simple_filename, boolean_filename, simple_size, boolean_size, data, simple_sampling = "level 2", boolean_sampling = "level 1",keys_and_counts = None, restrictions=None):
    #Using simple_filename generated from build_simple_file and
    #boolean_filename from build_boolean_file generates a list of NLI inpus
    #with simple_size simple examples and boolean_size compound examples
    #restrictions can be used to restrict the types of examples generated
    e,c,p = split_dict(simple_filename, restrictions)
    ekeys = list(e.keys())
    ckeys = list(c.keys())
    pkeys = list(p.keys())
    if simple_sampling == "level 0" or simple_sampling == "level 2" :
        ecounts, ccounts, pcounts = get_simple_encoding_counts(data, simple_sampling, ekeys, ckeys, pkeys)
    if simple_sampling == "level 1":
        ecounts = [1] * len(ekeys)
        ccounts = [1] * len(ckeys)
        pcounts = [1] * len(pkeys)
    label_size = int(simple_size/3)
    examples = []
    for i in range(label_size):
        encoding = json.loads(weighted_selection(ekeys, ecounts))
        premise, hypothesis = encoding_to_example(data,encoding)
        examples.append((premise.emptystring, "entailment", hypothesis.emptystring))
    for i in range(label_size):
        encoding = json.loads(weighted_selection(ckeys, ccounts))
        premise, hypothesis = encoding_to_example(data,encoding)
        examples.append((premise.emptystring, "contradiction", hypothesis.emptystring))
    for i in range(label_size):
        encoding = json.loads(weighted_selection(pkeys, pcounts))
        premise, hypothesis = encoding_to_example(data,encoding)
        examples.append((premise.emptystring, "neutral", hypothesis.emptystring))
    bool_label_size = int(boolean_size/3)
    bool_e,bool_c,bool_p = split_dict(boolean_filename, None)
    bool_ekeys = list(bool_e.keys())
    bool_ckeys = list(bool_c.keys())
    bool_pkeys = list(bool_p.keys())
    if keys_and_counts == None:
        keys_and_counts = sevenclass_simple_encodings(data,1, ekeys, ckeys, pkeys, ecounts, ccounts, pcounts)
    examples += generate_balanced_boolean_data(bool_ekeys, "entailment", keys_and_counts, boolean_sampling, bool_label_size, data)
    examples += generate_balanced_boolean_data(bool_ckeys, "contradiction", keys_and_counts, boolean_sampling, bool_label_size, data)
    examples += generate_balanced_boolean_data(bool_pkeys, "neutral", keys_and_counts, boolean_sampling, bool_label_size, data)
    random.shuffle(examples)
    return examples

def create_corpus(size, filename):
    data, _, _ = process_data(1.0)
    examples = generate_balanced_data("simple_solutions", "boolean_solutions", size, 0, data, simple_sampling = "level 2", boolean_sampling = "level 0")
    save_data(examples, filename)
if False:
    for ratio in [0,0.0625, 0.125, 0.25, 0.5, 0.75]:
        data, _, _ = process_data(1.0)
        restrictions, inverse_restrictions = nlm.create_gen_split(ratio)
        examples = generate_balanced_data("simple_solutions", "boolean_solutions", 500000, 0, data, simple_sampling = "level 2", boolean_sampling = "level 0",restrictions = restrictions)
        tr = examples
        save_data(examples, str(ratio) + "gendata.train")
        premise = parse_simple_sentence(data,tr[0][0])[0]
        hypothesis = parse_simple_sentence(data,tr[0][2])[0]
        _, relations_seen = nlm.compute_simple_relation_gentest(premise, hypothesis)
        for example in tr:
            premise = parse_simple_sentence(data,example[0])[0]
            hypothesis = parse_simple_sentence(data,example[2])[0]
            _, relations_seen = nlm.compute_simple_relation_gentest(premise, hypothesis, relations_seen)
        for k in relations_seen:
            print(k, len(relations_seen[k]))
            if len(relations_seen[k]) < 10:
                print(relations_seen[k])
        examples = generate_balanced_data("simple_solutions", "boolean_solutions", 10000, 0, data, simple_sampling = "level 2", boolean_sampling = "level 0",restrictions = inverse_restrictions)
        save_data(examples, str(ratio) +"gendata.val")
        examples = generate_balanced_data("simple_solutions", "boolean_solutions", 10000, 0, data, simple_sampling = "level 2", boolean_sampling = "level 0",restrictions = inverse_restrictions)
        save_data(examples, str(ratio) +"gendata.test")
