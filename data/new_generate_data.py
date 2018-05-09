import random
import json
import os
import copy
import pickle
from nltk.sem.logic import *
from nltk.inference import *
from nltk import Prover9
from nltk.corpus import wordnet
from joblib import Parallel, delayed

class sentence:
    #this class stores the logical representation of a sentence and the natural language representation of a sentence
    def __init__(self, core, passive, negate, adverb, adjective1, adjective2, determiners, adverb_position):
        self.core = core
        self.passive = passive
        self.negate = negate
        self.determiners = determiners
        self.negation = ["", ""]
        self.adverb = adverb 
        self.adjective1 = adjective1
        self.adjective2 = adjective2
        self.adverb_position = adverb_position
        self.negate_adverb = ""
        self.verb_adverb = ""
        self.sentence_adverb = ""
        if self.adverb_position == "before negation":
            assert self.adverb != ""
            assert negate == 1
            self.negate_adverb = self.adverb
            self.sentence_adverb = ""
            self.verb_adverb = ""
        if self.adverb_position == "before verb":
            assert self.adverb != ""
            self.verb_adverb = self.adverb
        if self.adverb_position == "end":
            self.sentence_adverb = self.adverb
            self.adverb_position = "end"
        self.non_passive_verb_index = 0
        if negate:
            self.negation = ["not", "does not"] #negation for passive and active
            self.non_passive_verb_index = 2#controls the verb form
        if self.passive:
            temp = self.adjective1
            self.adjective1 = self.adjective2#This ensures that identical adjectives appearing in
            self.adjective2 = temp           # both the premise and hypothesis apply to the same noun
            self.final = self.construct_string([self.determiners[0],self.adjective1,self.core[2],"is",self.negate_adverb,self.negation[0],self.verb_adverb,self.core[1][1],"by",self.determiners[1],self.adjective2,self.core[0],self.sentence_adverb])
        else:
            self.final = self.construct_string([self.determiners[0],self.adjective1,self.core[0],self.negate_adverb,self.negation[1],self.verb_adverb,self.core[1][self.non_passive_verb_index],self.determiners[1],self.adjective2,self.core[2],self.sentence_adverb])
        self.change_an() #replaces a with an if needed
        self.construct_logical_form()

    def change_an(self):
        #replace a with an where necessary
        if self.final[0] == "a" and self.final[2] in "aeiou":
            self.final = "an" + self.final[1:]
        for i in range(len(self.final) - 2):
            if self.final[i:i+3] == " a " and self.final[i+3] in "aeiou":
                self.final = self.final[:i+2] + "n" + self.final[i+2:]
                

    def construct_string(self,lst):
        #turn a list of words into a single sentence string
        result = ""
        for word in lst:
            if word != "":
                result += word + " "
        return result[:-1]

    def construct_logical_form(self):
        #construct a first order logic representation
        logical_form = ""
        append_formula = []
        working_core = list(copy.copy(self.core))
        verb = working_core[1][0]
        constant1 = "(constant1)"
        constant2 = "(constant2)"
        if self.passive: #ensures that subjects and objects have the same constants
            constant1 = "(constant2)"
            constant2 = "(constant1)"
        first_noun_arg = "(y)"
        second_noun_arg = "(x)"
        if self.determiners[1] == "the":
            first_noun_arg = constant1
        if self.determiners[0] == "the":
            second_noun_arg = constant2
        verb_arg = "(" + second_noun_arg[1:-1] + ","+ first_noun_arg[1:-1] + ")"
        if self.passive:
            working_core.reverse()
            verb_arg = "(" + first_noun_arg[1:-1] + ","+ second_noun_arg[1:-1] + ")"
        logical_form = verb + verb_arg
        if self.adverb_position == "before verb" or self.adverb_position == "end":
            logical_form = self.adverb + logical_form
            append_formula.append("all y.all x.(" + logical_form +"->" + verb + verb_arg + ")")
        temp = working_core[2] + first_noun_arg
        if self.adjective2 != "":
            temp = self.adjective2+working_core[2] + first_noun_arg
            append_formula.append("all x.(" + self.adjective2+working_core[2] + "(x)" +"->" + working_core[2] + "(x)" + ")")
        if self.determiners[1] == "every":
            logical_form = "all y.(" + temp + "->" + logical_form + ")"
        elif self.determiners[1] == "a":
            logical_form = "exists y.(" + temp + "&" + logical_form + ")"
        else:
            logical_form = "(" + temp + "&" + logical_form + ")"
        if self.negate:
            logical_form = "-" + logical_form
        if self.adverb_position == "before negation":  
            logical_form = "(" + logical_form + "&" + self.adverb+"not"+verb+ verb_arg + ")"
        temp = working_core[0] + second_noun_arg
        if self.adjective1 != "":
            temp = self.adjective1 + working_core[0] +second_noun_arg
            append_formula.append("all x.(" + self.adjective1+working_core[0] + "(x)" +"->" + working_core[0] + "(x)" + ")")
        if self.determiners[0] == "every":
            logical_form = "all x.(" + temp + "->" + logical_form + ")"
        elif self.determiners[0] == "a":
            logical_form = "exists x.(" + temp + "&" + logical_form + ")"
        else:
            logical_form = "(" + temp + "&" + logical_form + ")"
        logical_form = "(" + logical_form + ")"
        for form in append_formula:
            logical_form = logical_form + "&" + form
        self.logical_form = logical_form 


def process_data(train_ratio):
    #split the different parts of speech into train, validation, and test
    #determiners are not split
    train = dict()
    val = dict()
    test = dict()
    categories = ["agents", "transitive_verbs", "things", "determiners", "adverbs", "adjectives1","adjectives2"]
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

def get_cores(data, size=-1):
    #constructs a list of length size where each element
    cores = []
    if size != -1:
        counter = size
        while counter != 0:
            thing = data["things"][random.randint(0, len(data["things"]) - 1)]
            verb = data["transitive_verbs"][random.randint(0, len(data["transitive_verbs"])) - 1]
            agent = data["agents"][random.randint(0, len(data["agents"])) - 1]
            if (agent, verb, thing) not in cores:
                cores.append((agent, verb, thing))
                counter -= 1
    else:
        for thing in data["things"]:
            for agent in data["agents"]:
                for verb in data["transitive_verbs"]:
                    cores.append((agent, verb, thing))
    return cores

def get_word(data,chance, PoS, same=""):
    #if chance is 0 the empty string is returned
    #if chance is 1 and same is None then a random word of part of speech PoS is returned
    #otherwise there is a 0.5 chance that same is returned and an 0.5 chance that a random word of part of speech PoS is returned
    result = ""
    if random.randint(1,10) <= chance:
        result = random.choice(data[PoS])
    if same != "":
        temp = random.randint(1,3)
        if temp == 1:
            return same
        if temp == 2:
            return random.choice(data[PoS])
        if temp == 3:
            return ""
    return result

def get_label(prover, premise, hypothesis):
    #returns a label that is determined from using Prover9 on the first order logic representations
    p = Expression.fromstring(premise.logical_form)
    h = Expression.fromstring(hypothesis.logical_form)
    noth = Expression.fromstring("-" + hypothesis.logical_form)
    if prover.prove(h, [p]):
        return "entails"
    if prover.prove(noth, [p]):
        return "contradicts"
    return "permits"

def generate_random_example(inputs):
    #returns a tuple of premise string, label, hypothesis string, premise object, hypothesis object
    label, prover,data, core1, core2,passive, negate, adjective, adverb = inputs
    determiners = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
    adverb_word = get_word(data,adverb, "adverbs")
    adjective_word1 = get_word(data,adjective*6, "adjectives1")
    adjective_word2 = get_word(data,adjective*6, "adjectives2")
    passive_value = random.randint(0,passive)
    negation_value = random.randint(0,negate)
    adverb_location = "" 
    if adverb_word != "":
        adverb_location = ["end", "before verb"][random.randint(0,1)]
        if negation_value:
            adverb_location = ["end", "before verb", "before negation"][random.randint(0,2)]
    premise = sentence(core1, passive_value, negation_value, adverb_word, adjective_word1, adjective_word2, determiners, adverb_location)
    determiners = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
    adverb_word = get_word(data, adverb, "adverbs", adverb_word)
    adjective_word1 = get_word(data, adjective*5, "adjectives1", adjective_word1)
    adjective_word2 = get_word(data, adjective*5, "adjectives2", adjective_word2)
    passive_value = random.randint(0,passive)
    negation_value = random.randint(0,negate)
    adverb_location = "" 
    if adverb_word != "":
        adverb_location = ["end", "before verb"][random.randint(0,1)]
        if negation_value:
            adverb_location = ["end", "before verb", "before negation"][random.randint(0,2)]
    hypothesis = sentence(core2, passive_value, negation_value, adverb_word, adjective_word1, adjective_word2, determiners, adverb_location)
    if label == None:
        label = get_label(prover,premise, hypothesis)
    print(premise.final)
    return (premise.final, label, hypothesis.final, {"premise":[premise],"hypothesis":[hypothesis]})

def build_file(data):
    core = ["man", ["eats", "eaten", "eat"], "rock"]
    dets = ["a", "the", "every"]
    sentences = []
    encodings = []
    for pd1_index in range(3):
        pd1 = dets[pd1_index]
        for pd2_index in range(3):
            pd2 = dets[pd2_index]
            for hd1_index in range(3):
                hd1 = dets[hd1_index]
                for hd2_index in range(3):
                    hd2 = dets[hd2_index]
                    for ppassive_value in range(2):
                        for hpassive_value in range(2):
                            for padj1 in range(2):
                                if padj1 == 0:
                                    padj1_word = ""
                                else:
                                    padj1_word = random.choice(data["adjectives1"])
                                for hadj1 in range(2):
                                    if hadj1 == 0:
                                        hadj1_word = ""
                                    else:
                                        hadj1_word = ""
                                        while hadj1_word == padj1_word or hadj1_word == "":
                                            hadj1_word = random.choice(data["adjectives1"])
                                    for same_adj1 in range(2):
                                        if same_adj1 == 1 and (padj1 == 0 or hadj1 == 0):
                                            continue
                                        elif same_adj1 ==1:
                                            hadj1_word = padj1_word
                                        for padj2 in range(2):
                                            if padj2 == 0:
                                                padj2_word = ""
                                            else:
                                                padj2_word = random.choice(data["adjectives2"])
                                            for hadj2 in range(2):
                                                if hadj2 == 0:
                                                    hadj2_word = ""
                                                else:
                                                    hadj2_word = ""
                                                    while hadj2_word == padj2_word or hadj2_word == "":
                                                        hadj2_word = random.choice(data["adjectives2"])
                                                for same_adj2 in range(2):
                                                    if same_adj2 == 1 and (padj2 == 0 or hadj2 == 0):
                                                        continue
                                                    elif same_adj2 ==1:
                                                        hadj2_word = padj2_word
                                                    for padv in range(2):
                                                        if padv == 0:
                                                            padv_word = ""
                                                        else:
                                                            padv_word = random.choice(data["adverbs"])
                                                        for hadv in range(2):
                                                            if hadv == 0:
                                                                hadv_word = ""
                                                            else:
                                                                hadv_word = ""
                                                                while hadv_word == padv_word or hadv_word == "":
                                                                    hadv_word = random.choice(data["adverbs"])
                                                            for same_adv in range(2):
                                                                if same_adv == 1 and (padv == 0 or hadv == 0):
                                                                    continue
                                                                elif same_adv == 1:
                                                                    hadv_word = padv_word
                                                                for pnegation_value in range(2):
                                                                    for hnegation_value in range(2):
                                                                        for padv_location_index in range(4):
                                                                            padv_location = ["","before verb", "before negation", "end"][padv_location_index]
                                                                            if (padv_location_index == 2 and pnegation_value == 0) or (padv_location_index != 0 and padv ==0) or (padv_location_index == 0 and padv ==1):
                                                                                continue
                                                                            for hadv_location_index in range(4):
                                                                                hadv_location = ["","before verb", "before negation", "end"][hadv_location_index]
                                                                                if (hadv_location_index == 2 and hnegation_value == 0) or (hadv_location_index != 0 and hadv ==0) or (hadv_location_index == 0 and hadv ==1):
                                                                                    continue
                                                                                sentences.append([sentence(core, ppassive_value, pnegation_value, padv_word, padj1_word, padj2_word, [pd1,pd2], padv_location),sentence(core, hpassive_value, hnegation_value, hadv_word, hadj1_word, hadj2_word, [hd1,hd2], hadv_location)])
                                                                                encodings.append((ppassive_value, pnegation_value, padv, padj1, padj2, pd1_index,pd2_index, padv_location_index, hpassive_value, hnegation_value, hadv, hadj1, hadj2, hd1_index,hd2_index, hadv_location_index, same_adj1, same_adj2, same_adv))
    labels = Parallel(n_jobs=-1,backend="multiprocessing")(map(delayed(parallel_labels), sentences[:1]))
    result = dict()
    for i in range(len(labels)):
        result[json.dumps(encodings[i])] = labels[i]
    with open("big_data" + ".pkl", "w") as f:
        pickle.dump(json.dumps(result), f, pickle.HIGHEST_PROTOCOL)



global_prover = Prover9()
global_prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")

def parallel_labels(x):
    return get_label(global_prover, x[0], x[1])


def generate_examples(data, cores, passive, negate, adjective, adverb, distract):
    #returns a list of examples the same length as cores
    examples = []
    prover = Prover9()
    prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")
    cores = cores
    inputs = [[None, prover, data, core, core,passive, negate, adjective, adverb] for core in cores] #generate examples in parallel
    examples = Parallel(n_jobs=-1,backend="multiprocessing")(map(delayed(generate_random_example), inputs))
    random.shuffle(cores)
    for core in cores:
        distract -= 1
        if distract> 0:
            cores = [distraction(core, data), core]
            random.shuffle(cores)
            examples.append(generate_random_example(["permits", prover,data, cores[0], cores[1],passive, negate, adjective, adverb]))
    return examples

def distraction(core, data):
    #returns a tuple that has at least one element different from core
    result = [core[0], core[1], core[2]]
    inds = [0,1,2]
    ind = random.choice(inds)
    inds.remove(ind) 
    if ind ==0:
        temp = random.choice(data["agents"])
    if ind ==1:
        temp = random.choice(data["transitive_verbs"])
    if ind ==2:
        temp = random.choice(data["things"])
    result[ind] = temp    
    for _ in range(2):
        if random.randint(0,1):
            ind = random.choice(inds)
            inds.remove(ind)
            if ind ==0:
                temp = random.choice(data["agents"])
            if ind ==1:
                temp = random.choice(data["transitive_verbs"])
            if ind ==2:
                temp = random.choice(data["things"])
            result[ind] = temp    
    return tuple(result)

def build_boolean_file():
    logic_operators = ["|", "&", "->"]
    result = dict()
    for pindex in range(3):
        for hindex in range(3):
            for porder in range(2):
                horder = 0
                for first_relation in range(3):
                    for second_relation in range(3):
                        first_predicate = "A"
                        second_predicate = "B"
                        if porder:
                            temp = first_predicate
                            first_predicate = second_predicate
                            second_predicate = temp
                        first_assumption = "(" + first_predicate+"(constant)"+logic_operators[pindex] + second_predicate+"(constant)" + ")"
                        first_predicate = "C"
                        second_predicate = "D"
                        conclusion = "(" + first_predicate+"(constant)"+logic_operators[hindex] + second_predicate+"(constant)" + ")"
                        assumptions = [Expression.fromstring(first_assumption)]
                        if first_relation == 0:
                            assumptions.append(Expression.fromstring("A(constant)->C(constant)"))
                        if first_relation == 1:
                            assumptions.append(Expression.fromstring("-A(constant)|-C(constant)"))
                        if second_relation == 0:
                            assumptions.append(Expression.fromstring("B(constant)->D(constant)"))
                        if second_relation == 1:
                            assumptions.append(Expression.fromstring("-B(constant)|-D(constant)"))
                        label = None 
                        if global_prover.prove(Expression.fromstring(conclusion), assumptions):
                            label = "entails"
                        elif global_prover.prove(Expression.fromstring("-"+conclusion), assumptions):
                            label = "contradicts"
                        else:
                            label = "permits"
                        result[(pindex, hindex, porder, first_relation, second_relation)] = label
    with open("boolean_data" + ".pkl", "wb") as f:
        pickle.dump(str(result), f, pickle.HIGHEST_PROTOCOL)



def generate_compound(examples, first_predicate, second_predicate, sample, sample_index, conjunction, logic_operator):
    order = random.randint(0,1)
    if order:
        temp = first_predicate
        first_predicate = second_predicate
        second_predicate = temp
    sentence = sample[order][sample_index] + " " + conjunction + " " + sample[not order][sample_index]
    if conjunction == "then":
        sentence = "if " + sentence 
    logic_form = "(" + first_predicate+"(constant)"+logic_operator + second_predicate+"(constant)" + ")"
    return sentence, logic_form, order

def generate_boolean_examples(examples, size):
    prover = Prover9()
    prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")
    bool_examples = []
    conjunctions = ["or", "and", "then"]
    logic_operators = ["|", "&", "->"]
    for i in range(size):
        sample = random.sample(examples, 2)
        index = random.randint(0,2)
        prem_conjunction = conjunctions[index]
        logic_operator = logic_operators[index]
        premise, first_assumption, prem_order  = generate_compound(examples, "A", "B", sample, 0, prem_conjunction, logic_operator)
        index = random.randint(0,2)
        hyp_conjunction = conjunctions[index]
        logic_operator = logic_operators[index]
        hypothesis, conclusion, hyp_order = generate_compound(examples, "C", "D", sample, 2, hyp_conjunction, logic_operator)
        assumptions = [Expression.fromstring(first_assumption)]
        if sample[0][1] == "entails":
            assumptions.append(Expression.fromstring("A(constant)->C(constant)"))
        if sample[0][1] == "contradicts":
            assumptions.append(Expression.fromstring("-A(constant)|-C(constant)"))
        if sample[1][1] == "entails":
            assumptions.append(Expression.fromstring("B(constant)->D(constant)"))
        if sample[1][1] == "contradicts":
            assumptions.append(Expression.fromstring("-B(constant)|-D(constant)"))
        label = None 
        if prover.prove(Expression.fromstring(conclusion), assumptions):
            label = "entails"
        elif prover.prove(Expression.fromstring("-"+conclusion), assumptions):
            label = "contradicts"
        else:
            label = "permits"
        bool_examples.append((premise, label, hypothesis, {"premise":[sample[prem_order][3]["premise"][0],sample[not prem_order][3]["premise"][0]],"hypothesis":[sample[hyp_order][3]["hypothesis"][0],sample[not hyp_order][3]["hypothesis"][0]], "prem_conjunction":prem_conjunction, "hyp_conjunction":hyp_conjunction}))
    return bool_examples
    


def save_data(examples, name):
    data = []
    for example in examples:
        example_dict = dict()
        example_dict["sentence1"] = example[0]
        example_dict["sentence2"] = example[2]
        example_dict["gold_label"] = example[1]
        example_dict["example_data"] = example[3]
        data.append(example_dict)
    with open(name, 'wb') as f:
        pickle.dump(json.dumps(data), f, pickle.HIGHEST_PROTOCOL)

def restricted(restrictions, enc):
    for i in range(len(enc)):
        if restrictions[i] < enc[i]:
            return True
    return False 

def split_dict(filename, restrictions):
    with open(filename, 'rb') as f:
        stuff = pickle.load(f)
    stuff = json.loads(stuff)
    e = dict()
    c = dict()
    p = dict()
    for i in stuff:
        if restricted(restrictions,json.loads(i)):
            continue
        if stuff[i] == "entails":
            e[i] = stuff[i]
        if stuff[i] == "contradicts":
            c[i] = stuff[i]
        if stuff[i] == "permits":
            p[i] = stuff[i]
    return e, c, p

def bool_split_dict(filename, restrictions):
    with open(filename, 'rb') as f:
        stuff = pickle.load(f)
    stuff = eval(stuff)
    e = dict()
    c = dict()
    p = dict()
    for i in stuff:
        if restricted(restrictions,i):
            continue
        if stuff[i] == "entails":
            e[i] = stuff[i]
        if stuff[i] == "contradicts":
            c[i] = stuff[i]
        if stuff[i] == "permits":
            p[i] = stuff[i]
    return e, c, p

def encoding_to_example(data, enc, core1, core2):
    dets = ["a", "the", "every"]
    adv_locs = ["","before verb", "before negation", "end"]
    hadv_word = ""
    hadj1_word = ""
    hadj2_word = ""
    padv_word = ""
    padj1_word = ""
    padj2_word = ""
    if enc[2] == 1:
        padv_word = random.choice(data["adverbs"])
    if enc[3] == 1:
        padj1_word = random.choice(data["adjectives1"])
    if enc[4] == 1:
        padj2_word = random.choice(data["adjectives2"])
    if enc[10] == 1:
        hadv_word = padv_word
        if not enc[-1]:
            while hadv_word == padv_word:
                hadv_word = random.choice(data["adverbs"])
    if enc[11] == 1:
        hadj1_word = padj1_word
        if not enc[-3]:
            while hadj1_word == padj1_word:
                hadj1_word = random.choice(data["adjectives1"])
    if enc[12] == 1:
        hadj2_word = padj2_word
        if not enc[-2]:
            while hadj2_word == padj2_word:
                hadj2_word = random.choice(data["adjectives2"])
    return sentence(core1, enc[0], enc[1], padv_word, padj1_word, padj2_word, [dets[enc[5]],dets[enc[6]]], adv_locs[enc[7]]),sentence(core2, enc[8], enc[9], hadv_word,hadj1_word, hadj2_word, [dets[enc[13]],dets[enc[14]]], adv_locs[enc[15]])


def generate_balanced_data(filename, boolfilename, size, boolean_size, cores, data, restrictions=[1000000]*19):
    e,c,p = split_dict(filename, restrictions)
    ekeys = list(e.keys())
    ckeys = list(c.keys())
    pkeys = list(p.keys())
    print(len(ekeys), len(ckeys), len(pkeys))
    allkeys =  ekeys + ckeys + pkeys
    distractions = []
    for k in allkeys:
        distractions.append(("distract", k))
    label_size = int(size/3)
    examples = []
    for i in range(label_size):    
        encoding = json.loads(random.choice(ekeys))
        core = cores[i%len(cores)]
        premise, hypothesis = encoding_to_example(data,encoding, core,core)
        examples.append((premise.final, "entails", hypothesis.final, [(encoding,core,core)]))
    for i in range(label_size):    
        encoding = json.loads(random.choice(ckeys))
        core = cores[(i+label_size)%len(cores)]
        premise, hypothesis = encoding_to_example(data,encoding, core, core)
        examples.append((premise.final, "contradicts", hypothesis.final, [(encoding,core,core)]))
    for i in range(label_size):    
        choice = random.choice(list(pkeys) + distractions)
        core1 = cores[(i+label_size*2)%len(cores)]
        encoding = None
        core2 = core1 
        if isinstance(choice, tuple):
            encoding = json.loads(choice[1])
            core2 = distraction(core,data)
        else:
            encoding = json.loads(choice)
        premise, hypothesis = encoding_to_example(data,encoding, core1, core2)
        examples.append((premise.final, "permits", hypothesis.final, [(encoding,core1,core2)] ))
    bool_label_size = int(boolean_size/3)
    bool_e,bool_c,bool_p = bool_split_dict(boolfilename, restrictions)
    bool_ekeys = list(bool_e.keys())
    bool_ckeys = list(bool_c.keys())
    bool_pkeys = list(bool_p.keys())
    for i in range(bool_label_size):
        encoding = random.choice(bool_ekeys)
        pcore = cores[(i)%len(cores)]
        pcore2 = pcore
        if encoding[3] == 0:
            simple1_encoding = json.loads(random.choice(ekeys))
            premise1, hypothesis1 = encoding_to_example(data, simple1_encoding, pcore, pcore2)
        if encoding[3] == 1:
            simple1_encoding = json.loads(random.choice(ckeys))
            premise1, hypothesis1 = encoding_to_example(data, simple1_encoding, pcore, pcore2)
        if encoding[3] == 2:
            choice = random.choice(list(pkeys) + distractions)
            if isinstance(choice, tuple):
                simple1_encoding = json.loads(choice[1])
                pcore2 = distraction(pcore,data)
            else:
                simple1_encoding = json.loads(choice)
            premise1, hypothesis1 = encoding_to_example(data,simple1_encoding, pcore, pcore2)
        hcore = cores[(i + random.randint(1,len(cores)-1))%len(cores)]
        hcore2 = hcore
        if encoding[4] == 0:
            simple2_encoding = json.loads(random.choice(ekeys))
            premise2, hypothesis2 = encoding_to_example(data, simple2_encoding, hcore, hcore2)
        if encoding[4] == 1:
            simple2_encoding = json.loads(random.choice(ckeys))
            premise2, hypothesis2 = encoding_to_example(data, simple2_encoding, hcore, hcore2)
        if encoding[4] == 2:
            choice = random.choice(list(pkeys) + distractions)
            if isinstance(choice, tuple):
                simple2_encoding = json.loads(choice[1])
                hcore2 = distraction(hcore,data)
            else:
                simple2_encoding = json.loads(choice)
            premise2, hypothesis2 = encoding_to_example(data,simple2_encoding, hcore, hcore2)
        if encoding[2] == 1:
            temp = premise2
            premise2 = premise1
            premise1 = temp
        conjunctions = ["or", "and", "then"]
        premise_conjunction = conjunctions[encoding[0]]
        hypothesis_conjunction = conjunctions[encoding[1]]
        premise_compound = premise1.final + " " + premise_conjunction + " " + premise2.final
        hypothesis_compound = hypothesis1.final+ " " + hypothesis_conjunction+ " " + hypothesis2.final
        if premise_conjunction == "then":
            premise_compound = "if " + premise_compound
        if hypothesis_conjunction == "then":
            hypothesis_compound = "if " + hypothesis_compound
        examples.append((premise_compound, "entails", hypothesis_compound, [(simple1_encoding, pcore, pcore2), (simple2_encoding, hcore, hcore2), (encoding,)]))
    for i in range(bool_label_size):
        encoding = random.choice(bool_ckeys)
        pcore = cores[(i)%len(cores)]
        pcore2 = pcore
        if encoding[3] == 0:
            simple1_encoding = json.loads(random.choice(ekeys))
            premise1, hypothesis1 = encoding_to_example(data, simple1_encoding, pcore, pcore2)
        if encoding[3] == 1:
            simple1_encoding = json.loads(random.choice(ckeys))
            premise1, hypothesis1 = encoding_to_example(data, simple1_encoding, pcore, pcore2)
        if encoding[3] == 2:
            choice = random.choice(list(pkeys) + distractions)
            if isinstance(choice, tuple):
                simple1_encoding = json.loads(choice[1])
                pcore2 = distraction(pcore,data)
            else:
                simple1_encoding = json.loads(choice)
            premise1, hypothesis1 = encoding_to_example(data,simple1_encoding, pcore, pcore2)
        hcore = cores[(i + random.randint(1,len(cores)-1))%len(cores)]
        hcore2 = hcore
        if encoding[4] == 0:
            simple2_encoding = json.loads(random.choice(ekeys))
            premise2, hypothesis2 = encoding_to_example(data, simple2_encoding, hcore, hcore2)
        if encoding[4] == 1:
            simple2_encoding = json.loads(random.choice(ckeys))
            premise2, hypothesis2 = encoding_to_example(data, simple2_encoding, hcore, hcore2)
        if encoding[4] == 2:
            choice = random.choice(list(pkeys) + distractions)
            if isinstance(choice, tuple):
                simple2_encoding = json.loads(choice[1])
                hcore2 = distraction(hcore,data)
            else:
                simple2_encoding = json.loads(choice)
            premise2, hypothesis2 = encoding_to_example(data,simple2_encoding, hcore, hcore2)
        if encoding[2] == 1:
            temp = premise2
            premise2 = premise1
            premise1 = temp
        conjunctions = ["or", "and", "then"]
        premise_conjunction = conjunctions[encoding[0]]
        hypothesis_conjunction = conjunctions[encoding[1]]
        premise_compound = premise1.final + " " + premise_conjunction + " " + premise2.final
        hypothesis_compound = hypothesis1.final+ " " + hypothesis_conjunction+ " " + hypothesis2.final
        if premise_conjunction == "then":
            premise_compound = "if " + premise_compound
        if hypothesis_conjunction == "then":
            hypothesis_compound = "if " + hypothesis_compound
        examples.append((premise_compound, "contradicts", hypothesis_compound, [(simple1_encoding, pcore, pcore2), (simple2_encoding, hcore, hcore2), (encoding,)]))
    for i in range(bool_label_size):
        encoding = random.choice(bool_ekeys)
        pcore = cores[(i)%len(cores)]
        pcore2 = pcore
        if encoding[3] == 0:
            simple1_encoding = json.loads(random.choice(ekeys))
            premise1, hypothesis1 = encoding_to_example(data, simple1_encoding, pcore, pcore2)
        if encoding[3] == 1:
            simple1_encoding = json.loads(random.choice(ckeys))
            premise1, hypothesis1 = encoding_to_example(data, simple1_encoding, pcore, pcore2)
        if encoding[3] == 2:
            choice = random.choice(list(pkeys) + distractions)
            if isinstance(choice, tuple):
                simple1_encoding = json.loads(choice[1])
                pcore2 = distraction(pcore,data)
            else:
                simple1_encoding = json.loads(choice)
            premise1, hypothesis1 = encoding_to_example(data,simple1_encoding, pcore, pcore2)
        hcore = cores[(i + random.randint(1,len(cores)-1))%len(cores)]
        hcore2 = hcore
        if encoding[4] == 0:
            simple2_encoding = json.loads(random.choice(ekeys))
            premise2, hypothesis2 = encoding_to_example(data, simple2_encoding, hcore, hcore2)
        if encoding[4] == 1:
            simple2_encoding = json.loads(random.choice(ckeys))
            premise2, hypothesis2 = encoding_to_example(data, simple2_encoding, hcore, hcore2)
        if encoding[4] == 2:
            choice = random.choice(list(pkeys) + distractions)
            if isinstance(choice, tuple):
                simple2_encoding = json.loads(choice[1])
                hcore2 = distraction(hcore,data)
            else:
                simple2_encoding = json.loads(choice)
            premise2, hypothesis2 = encoding_to_example(data,simple2_encoding, hcore, hcore2)
        if encoding[2] == 1:
            temp = premise2
            premise2 = premise1
            premise1 = temp
        conjunctions = ["or", "and", "then"]
        premise_conjunction = conjunctions[encoding[0]]
        hypothesis_conjunction = conjunctions[encoding[1]]
        premise_compound = premise1.final + " " + premise_conjunction + " " + premise2.final
        hypothesis_compound = hypothesis1.final+ " " + hypothesis_conjunction+ " " + hypothesis2.final
        if premise_conjunction == "then":
            premise_compound = "if " + premise_compound
        if hypothesis_conjunction == "then":
            hypothesis_compound = "if " + hypothesis_compound
        examples.append((premise_compound, "permits", hypothesis_compound, [(simple1_encoding, pcore, pcore2), (simple2_encoding, hcore, hcore2), (encoding,)]))
    random.shuffle(examples)
    return examples

def check_data(data):
    for k in data:
        result = set()
        for i in range(len(data[k])):
            for j in range(i+1, len(data[k])):
                if data[k][i] == data[k][j]:
                    print(data[k][i])
    for k in ["transitive_verbs"]:#data:
        result = set()
        x = copy.copy(data[k])
        for i in range(len(data[k])):
            w = data[k][i]
            if k == "transitive_verbs":
                w = w[2]
            for j in range(i+1, len(data[k])):
                w2 = data[k][j]
                if k == "transitive_verbs":
                    w2 = w2[2]
                if w == w2:
                    continue
                a = wordnet.synsets(w)
                b = wordnet.synsets(w2)
                for t in a:
                    for s in b:
                        if t in s.hypernyms() or s in t.hypernyms():
                            result.add((w,w2))
                        for q in t.lemmas():
                            for e in s.lemmas():
                                if q in e.antonyms() or e in q.antonyms():
                                    result.add((w,w2))
        print(result)



if __name__ == "__main__":
    #encodings.append([pindex, hindex, porder, horder,first_relation, second_relation])
    build_boolean_file()
    data, _, _ = process_data(1.0)
    check_data(data)
     
    cores = get_cores(data)
    size = 2000000
    examples = generate_balanced_data("big_data.pkl", "boolean_data.pkl",6,50 , cores, data)
    save_data(examples[:int(size*0.9)], "simplejoint.train")
    save_data(examples[int(size*0.9):int(size*0.95)], "simplejoint.val")
    save_data(examples[int(size*0.95):], "simplejoint.test")
    train_data, val_data, test_data = process_data(0.6)
    train_cores = get_cores(train_data)
    val_cores = get_cores(val_data)
    test_cores = get_cores(test_data)
    size = 2000000
    train_examples = generate_balanced_data("big_data.pkl","boolean_data.pkl",2000000, 0, cores, train_data)
    val_examples = generate_balanced_data("big_data.pkl","boolean_data.pkl",2000000, 0, cores, val_data)
    test_examples = generate_balanced_data("big_data.pkl","boolean_data.pkl",2000000, 0, cores, test_data)
    save_data(examples[:int(size*0.9)], "simpledisjoint.train")
    save_data(examples[int(size*0.9):int(size*0.95)], "simpledisjoint.val")
    save_data(examples[int(size*0.95):], "simpledisjoint.test")
