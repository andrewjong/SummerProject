import pickle
import nltk
import os
from sklearn import linear_model
from sklearn import metrics

class sentence:
    def __init__(self, core, passive, negate, adverb, data, determiners):
        self.core = core
        self.passive = passive
        self.negate = negate
        self.determiners = determiners
        self.negation = ["", ""]
        self.adverb = adverb + " "
        self.adverb_present = ""
        self.negate_adverb = ""
        self.verb_adverb = ""
        self.sentence_adverb = ""
        adj_place = random.randint(0,1)
        if adj_place == 1 and self.adverb != "":
            self.verb_adverb = self.adverb
            self.adverb_present = "normal"
        elif self.adverb != "":
            self.sentence_adverb = self.adverb
            self.adverb_present = "normal"
        adj_place = random.randint(0,1)
        if adj_place == 1 and self.negate:
            self.negate_adverb = self.adverb
            self.adverb_present = "negate"
            self.sentence_adverb = ""
            self.verb_adverb = ""
        self.non_passive_verb_index = 0
        if negate:
            self.negation = ["not ", "does not "]
            self.non_passive_verb_index = 2
        if negate and self.determiners in [("every","every"),("a", "a")]:
            self.uhoh = True
        elif not negate and self.determiners in [("every","a"), ("a","every")]:
            self.uhoh = True
        else:
            self.uhoh =False
        if self.passive:
            self.final = self.determiners[0] + " " + self.core[2] + " "  + "is" + " " + self.negate_adverb + self.negation[0]  +self.verb_adverb  +self.core[1][1] + " " + "by" + " " + self.determiners[1]  + " " + self.core[0] +" " + self.sentence_adverb
        else:
            self.final = self.determiners[0] + " " + self.core[0] + " " + self.negate_adverb + self.negation[1]  +self.verb_adverb +self.core[1][self.non_passive_verb_index] + " "  +self.determiners[1] + " "  + self.core[2] + " " +self.sentence_adverb

def log_form(s1, s2,p,h, words):
    result = []
    p_list = str.split(s1)
    h_list = str.split(s2)
    result.append(nltk.translate.bleu_score.sentence_bleu([p_list], h_list))
    result.append(nltk.translate.bleu_score.sentence_bleu([p_list], h_list, weights = (1,)))
    result.append(nltk.translate.bleu_score.sentence_bleu([p_list], h_list, weights = (0.5,0.5)))
    result.append(nltk.translate.bleu_score.sentence_bleu([p_list], h_list, weights = (1.0/3.0,1.0/3.0,1.0/3.0)))
    result.append(len(s1) -len(s2))
    count = 0
    for pword in p_list:
        for hword in h_list:
            if pword == hword:
                count += 1
                break
    result.append(count)
    result.append(count/min(len(p_list), len(h_list)))
    noun = 0
    verb = 0
    adverb = 0 
    padv = 0
    hadv = 0
    count = len(p)
    for s in p:
        if s.adverb != " ":
            padv +=1
        for s2 in h:
            if s2.adverb != " ":
                hadv +=1
            if s.core[0] == s2.core[0]:
                noun +=1
            if s.core[2] == s2.core[2]:
                noun +=1
            if s.core[1] == s2.core[1]:
                verb +=1
            if s2.adverb == s.adverb and s2.adverb != " ":
                adverb +=1
    result.append(noun)
    result.append(verb)
    result.append(noun/(count*2))
    result.append(verb/count)
    result.append(adverb)
    result.append(adverb/max(min(padv, hadv),1))
    uni = [0] * 562
    bi = [0] * (562 * 562)
    for word in h_list:
        uni[words.index(word)] = 1
    for gram in nltk.bigrams(h_list):
        bi[words.index(gram[0])*556 + words.index(gram[1])] = 1
    result = result + uni + bi
    uni = [0] * 562
    bi = [0] * (562 * 562)
    for word in p_list:
        uni[words.index(word)] = 1
    for gram in nltk.bigrams(p_list):
        bi[words.index(gram[0])*562 + words.index(gram[1])] = 1
    result = result + uni + bi
    bi = [0] * (562 * 562)
    for word1 in p_list:
        for word2 in h_list:
            bi[words.index(word1)*562 + words.index(word2)] = 1
    result = result + bi
    return result

def make_data(name):
    data = []
    label = []
    examples = pickle.load(open("data/TrainingData/stuff." + name + ".pkl","rb"))
    with open(os.path.join("data\Data", "agents.txt"),"r") as f:
        agents = f.readlines()
        agents = [agent.strip() for agent in agents]
    with open(os.path.join("data\Data", "transitive_verbs.txt"),"r") as f:
        transitive_verbs = f.readlines()
        transitive_verbs = [verb.strip().split() for verb in transitive_verbs]
        verbs = []
        for v in transitive_verbs:
            verbs = verbs + v
    with open(os.path.join("data\Data", "things.txt"),"r") as f:
        things = f.readlines()
        things = [thing.strip() for thing in things]
    with open(os.path.join("data\Data", "determiners.txt"),"r") as f:
        determiners= f.readlines()
        determiners= [determiner.strip() for determiner in determiners]
    with open(os.path.join("data\Data", "adverbs.txt"),"r") as f:
        adverbs = f.readlines()
        adverbs = [adverb.strip() for adverb in adverbs]
    words = agents+things + verbs + ["is", "does"] + determiners + adverbs + ["by", "not", "if", "then", "and", "or"]
    for example in examples:
        s1 = example[0]
        s2 = example[2]
        p = example[3]
        h = example[4]
        if not isinstance(p, tuple):
            p = [p]
        if not isinstance(h, tuple):
            h = [h]
        data.append(log_form(s1,s2,p,h, words))
        if example[1] =="entails":
            label.append(0)
        if example[1] == "contradicts":
            label.append(1)
        if example[1] == "permits":
            label.append(2)
    with open("unlex."+ name+ '.pkl', 'wb') as f:
        pickle.dump((data,label), f, pickle.HIGHEST_PROTOCOL)
    return data, label

trainx, trainy = make_data("train")
testx, testy = make_data("test")
jtestx, jtesty = make_data("jointtest")
valx, valy = make_data("val")

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(trainx, trainy)

print(metrics.accuracy_score(trainy, mul_lr.predict(trainx)))
print(metrics.accuracy_score(valy, mul_lr.predict(valx)))
print(metrics.accuracy_score(testy, mul_lr.predict(testx)))
print(metrics.accuracy_score(jtesty, mul_lr.predict(jtestx)))
