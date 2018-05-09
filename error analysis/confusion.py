import numpy
import pickle
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
def confuse(name):
    conf = numpy.zeros((3,3))
    with open(name + '.pkl', 'rb') as f:
        stuff = pickle.load(f)
    for pred, label in stuff:
        conf[pred][label] += 1
    return conf/numpy.sum(conf)

def anal(name):
    with open(name + '.pkl', 'rb') as f:
        stuff = pickle.load(f)
    with open("stuff." + name + '.pkl', 'rb') as f:
        stuff2 = pickle.load(f)
    print(len(stuff))
    print(len(stuff2))
    yes = dict()
    no = dict()
    for i in range(len(stuff)):
        if stuff[i][0] == stuff[i][1]:
            if stuff2[i][-1] in yes:
                yes[stuff2[i][-1]] += 1
            else:
                yes[stuff2[i][-1]] = 1
        else:
            if stuff2[i][-1] in no:
                no[stuff2[i][-1]] += 1
            else:
                no[stuff2[i][-1]] = 1
    return yes, no

jt = anal("jointtest")
for k in jt[0]:
    print(k)
    print(jt[0][k])
    print(jt[1][k])
    print(jt[0][k]/(jt[0][k] + jt[1][k]))
t = anal("test")
for k in t[0]:
    print(k)
    print(t[0][k])
    print(t[1][k])
    print(t[0][k]/(t[0][k] + t[1][k]))
tr = anal("train")

