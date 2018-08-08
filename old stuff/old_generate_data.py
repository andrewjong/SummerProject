import random
import os
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

def process_data():
    ratio = 0.6
    with open(os.path.join("data", "agents.txt"),"r") as f:
        agents = f.readlines()
        agents = [agent.strip() for agent in agents]
    with open(os.path.join("data", "transitive_verbs.txt"),"r") as f:
        transitive_verbs = f.readlines()
        transitive_verbs = [verb.strip().split() for verb in transitive_verbs]
    with open(os.path.join("data", "things.txt"),"r") as f:
        things = f.readlines()
        things = [thing.strip() for thing in things]
    with open(os.path.join("data", "determiners.txt"),"r") as f:
        determiners= f.readlines()
        determiners= [determiner.strip() for determiner in determiners]
    with open(os.path.join("data", "adverbs.txt"),"r") as f:
        adverbs = f.readlines()
        adverbs = [adverb.strip() for adverb in adverbs]
    print(len(agents) * len(things) * len(transitive_verbs))
    random.shuffle(agents)
    random.shuffle(things)
    random.shuffle(transitive_verbs)
    random.shuffle(determiners)
    random.shuffle(adverbs)
    return {"things": things[:int(len(things)*ratio)], "transitive_verbs": transitive_verbs[:int(len(transitive_verbs)*ratio)], "agents":agents[:int(len(agents)*ratio)], "adverbs":adverbs[:int(len(adverbs)*ratio)], "determiners":determiners}, {"things": things[int(len(things)*ratio):int(len(things)*((1-ratio)*0.5 + ratio))], "transitive_verbs": transitive_verbs[int(len(transitive_verbs)*ratio):int(len(transitive_verbs)*((1-ratio)*0.5 + ratio))], "agents":agents[int(len(agents)*ratio):int(len(agents)*((1-ratio)*0.5+ratio))],"adverbs":adverbs[int(len(adverbs)*ratio):int(len(adverbs)*((1-ratio)*0.5+ratio)) ], "determiners":determiners}, {"things": things[int(len(things)*((1-ratio)*0.5+ratio)):], "transitive_verbs": transitive_verbs[int(len(transitive_verbs)*((1-ratio)*0.5+ratio)):], "agents":agents[int(len(agents)*((1-ratio)*0.5 + ratio)):],"adverbs":adverbs[int(len(adverbs)*((1-ratio)*0.5 + ratio)):], "determiners":determiners}, 

def get_cores(data, size):
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

def generate_examples(data, cores, passive, negate, adj, distract, ):
    examples = []
    for core in cores:
        passive_premise = random.randint(0,passive)
        negate_premise = random.randint(0,negate)
        passive_hypothesis = random.randint(0,passive)
        adverb_premise= random.randint(0,adj)
        negate_hypothesis = random.randint(0,negate)
        negate_hypothesis = random.randint(0,negate)
        adverb_hypothesis = random.randint(0,adj)
        adverbh = ""
        adverbp = ""
        if adverb_premise:
            adverbp = random.choice(data["adverbs"])
        if adverb_hypothesis:
            same = random.randint(0,1)
            if same and adverb_premise:
                adverbh = adverbp
            else:
                while adverbh != adverbp:
                    adverbh = random.choice(data["adverbs"])
        pd = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
        hd = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
        premise = sentence(core, passive_premise, negate_premise, adverbp, data, pd)
        hypothesis = sentence(core, passive_hypothesis, negate_hypothesis, adverbh, data, hd)
        label = get_label(premise, hypothesis)
        #print(premise.final)
        #print(label)
        #print(hypothesis.final)
        examples.append((premise.final, label, hypothesis.final, premise,hypothesis, ""))
    count = 0
    for core in cores:
        count += 1
        if count < distract:
            cores = [distraction(core, data), core]
            random.shuffle(cores)
            passive_premise = random.randint(0,passive)
            negate_premise = random.randint(0,negate)
            adverb_premise = random.randint(0,adj)
            passive_hypothesis = random.randint(0,passive)
            negate_hypothesis = random.randint(0,negate)
            adverb_hypothesis = random.randint(0,adj)
            adverbh = ""
            adverbp = ""
            if adverb_premise:
                adverbp = random.choice(data["adverbs"])
            if adverb_hypothesis:
                same = random.randint(0,1)
                if same and adverb_premise:
                    adverbh = adverbp
                else:
                    while adverbh != adverbp:
                        adverbh = random.choice(data["adverbs"])
            pd = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
            hd = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
            premise = sentence(cores[0], passive_premise, negate_premise, adverbp, data, pd)
            hypothesis = sentence(cores[1], passive_hypothesis, negate_hypothesis, adverbh, data, hd)
            examples.append((premise.final, "permits",  hypothesis.final, premise, hypothesis, ""))
    return examples

def distraction(core, data):
    result = [core[0], core[1], core[2]]
    inds = [0,1,2]
    ind = random.choice(inds)
    inds.remove(ind)
    while tuple(result) == core:
        if ind ==0:
            temp = random.choice(data["agents"])
        if ind ==1:
            temp = random.choice(data["transitive_verbs"])
        if ind ==2:
            temp = random.choice(data["things"])
        result[ind] = temp    
    again = random.randint(1,10)
    if again <=3:
        ind = random.choice(inds)
        inds.remove(ind)
        if ind ==0:
            temp = random.choice(data["agents"])
        if ind ==1:
            temp = random.choice(data["transitive_verbs"])
        if ind ==2:
            temp = random.choice(data["things"])
        result[ind] = temp    
    again = random.randint(1,10)
    if again <=3:
        ind = random.choice(inds)
        if ind ==0:
            temp = random.choice(data["agents"])
        if ind ==1:
            temp = random.choice(data["transitive_verbs"])
        if ind ==2:
            temp = random.choice(data["things"])
        result[ind] = temp    
    return tuple(result)


def human_test(data, core, passive, negate):
    examples = []
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                for l in range(0,2):
                    for d1 in ["a", "the","every"]:
                        for d2 in ["a", "the", "every"]:
                            for d3 in ["a", "the", "every"]:
                                for d4 in ["a", "the", "every"]:
                                    passive_premise = i  
                                    negate_premise = j
                                    passive_hypothesis = k
                                    negate_hypothesis = l
                                    premise = sentence(core, passive_premise, negate_premise, data, (d1,d2))
                                    hypothesis = sentence(core, passive_hypothesis, negate_hypothesis, data, (d3,d4))
                                    label = get_label(premise, hypothesis)
                                    examples.append((premise.final, label, hypothesis.final, premise, hypothesis))
    for e in examples:
        print(e)
    return examples


def generate_boolean_examples(examples, size):
    bool_examples = []
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        label = "permits"
        if sample[0][1] == "contradicts" or sample[1][1] == "contradicts":
            label = "contradicts"
        if sample[0][1] == "entails" and sample[1][1] == "entails":
            label = "entails"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            label = "permits"
            if sample[0][1] == "contradicts" or sample[1][1] == "contradicts":
                label = "contradicts"
            if sample[0][1] == "entails" and sample[1][1] == "entails":
                label = "entails"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        premise = sample[order1][0] + " and " + sample[not order1][0]
        hypothesis = sample[order2][2] + " and " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[0][3], sample[1][3]),(sample[0][4], sample[1][4]), "and" ))
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
            label = "contradicts"
        if sample[0][1] == "entails" and sample[1][1] == "entails":
            label = "entails"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
                label = "contradicts"
            if sample[0][1] == "entails" and sample[1][1] == "entails":
                label = "entails"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        premise = sample[order1][0] + " or " + sample[not order1][0]
        hypothesis = sample[order2][2] + " or " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[0][3], sample[1][3]),(sample[0][4], sample[1][4]), "or" ))
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        if sample[0][1] == "entails" and sample[1][1] == "contradicts":
            label = "contradicts"
        if sample[0][1] == "entails" and sample[1][1] == "entails":
            label = "entails"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            if sample[0][1] == "entails" and sample[1][1] == "contradicts":
                label = "contradicts"
            if sample[0][1] == "entails" and sample[1][1] == "entails":
                label = "entails"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        if label != "permits" or sample[0][1] == "contradicts" and sample[1][1] == "entails":
            order1 = 0
            order2 = 0
        premise = "if " + sample[order1][2] + " then " + sample[not order1][0]
        hypothesis = "if " + sample[order2][0] + " then " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[order1][3], sample[not order1][3]),(sample[order2][4], sample[not order2][4]), "ifthen" ))
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        if sample[0][1] == "contradicts" and sample[1][1] == "entails":
            label = "entails"
        if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
            label = "contradicts"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            if sample[0][1] == "contradicts" and sample[1][1] == "entails":
                label = "entails"
            if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
                label = "contradicts"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        if label != "permits" or sample[0][1] == "entails" and sample[1][1] == "contradicts":
            order2 = 0
        premise = sample[order1][0] + " or " + sample[not order1][0]
        hypothesis = "if " + sample[order2][2] + " then " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[order1][3], sample[not order1][3]),(sample[order2][4], sample[not order2][4]), "ifthenor" ))
    return bool_examples
    

def get_label(premise, hypothesis):
    contradiction_agent = [("a", "every"), ("the", "the"), ("the", "every"), ("every", "every"), ("every", "a"), ("every", "the")]
    negated_thing= [("every", "every"), ("the", "the"), ("the", "every"), ("a", "every"), ("a", "a"), ("a", "the")]
    entailment_agent = [("a", "a"), ("the", "the"), ("every", "every"), ("the", "a"), ("every", "a"), ("every", "the")]
    output = ""
    skip = False
    if premise.passive != hypothesis.passive:
        if not premise.uhoh:
            change_passive(premise)
        elif not hypothesis.uhoh:
            change_passive(hypothesis)
        if premise.uhoh and hypothesis.uhoh:
            if premise.negate and premise.determiners == ("a","a") and not hypothesis.negate and hypothesis.determiners == ("a", "every"):
                output = "contradicts"
                skip = True
            elif hypothesis.negate and hypothesis.determiners == ("a","a") and not premise.negate and premise.determiners == ("a", "every"):
                output = "contradicts"
                skip = True
            else:
                skip = True
                output =  "permits"
    if not skip:
        if premise.negate != hypothesis.negate:
            if premise.negate and (premise.determiners[0], hypothesis.determiners[0]) in contradiction_agent and (premise.determiners[1], hypothesis.determiners[1]) in negated_thing:
                output = "contradicts"
            if hypothesis.negate and (hypothesis.determiners[0], premise.determiners[0]) in contradiction_agent and (hypothesis.determiners[1], premise.determiners[1]) in negated_thing:
                output = "contradicts"
            else:
                output = "permits"
        else:
            if (premise.determiners[0], hypothesis.determiners[0]) in entailment_agent and ((premise.negate and (premise.determiners[1], hypothesis.determiners[1]) in negated_thing)or (not premise.negate and (premise.determiners[1], hypothesis.determiners[1]) in entailment_agent)):
                output = "entails"
            else:
                output = "permits"
    if output == "entails" and not premise.negate:
        if premise.adverb == "" and hypothesis.adverb_present == "normal":
            output = "permits"
        if premise.adverb_present == "normal" and hypothesis.adverb=="normal" and premise.adverb != hypothesis.adverb:
            output = "permits"
    if output == "entails" and premise.negate:
        if premise.adverb == "" and hypothesis.adverb_present == "negate":
            output = "permits"
        if premise.adverb_present == "negate" and hypothesis.adverb=="negate" and premise.adverb != hypothesis.adverb:
            output = "permits"
        if premise.adverb_present == "normal" and (hypothesis.adverb_present == "" or hypothesis.adverb_present == "negate"):
            output = "permits"
        if premise.adverb_present == "normal" and hypothesis.adverb_present == "normal"  and hypothesis.adverb != premise.adverb:
            output = "permits"
    if output == "contradicts":
        n = ""
        p = ""
        if premise.negate:
            n = premise
            p = hypothesis
        else:
            n = hypothesis
            p = premise
        if n.adverb_present == "normal" and p.adverb_present == "":
            output = "permits"
        if n.adverb_present == "normal" and p.adverb_present == "normal" and p.adverb != n.adverb:
            output = "permits"
    return output

def change_passive(s):
    new = []
    if s.negate:
        if s.determiners[1] == "a":
            new.append("every")
        elif s.determiners[1] == "every":
            new.append("a")
        else:
            new.append("the")
        new.append(s.determiners[0])
    else:
        new = (s.determiners[1], s.determiners[0])
    s.determiners = tuple(new)
        

def save_data(examples, name):
    prem = []
    hyp = []
    label = []
    stuff = []
    for example in examples:
        prem.append(example[0] + "\n")
        label.append(example[1] + "\n")
        hyp.append(example[2] + "\n")
        stuff.append(example)
    with open("pi." + "prem." + name, "w") as f:
        f.writelines(prem)
    with open("pi." + "hyp." + name, "w") as f:
        f.writelines(hyp)
    with open("pi." + "label." + name, "w") as f:
        f.writelines(label)
    with open("stuff."+ name+ '.pkl', 'wb') as f:
        pickle.dump(stuff, f, pickle.HIGHEST_PROTOCOL)



train_data, validation_data, test_data = process_data()
print(train_data)
print(test_data)
train_cores = get_cores(train_data,-1)
validation_cores = get_cores(validation_data,-1)
test_cores = get_cores(test_data,-1)
#human_test(test_data, test_cores[0],1,1)
if True:
    train_examples = generate_examples(train_data, train_cores,1,1,1, len(train_cores)/5)
    validation_examples = generate_examples(validation_data, validation_cores,1,1,1,len(validation_cores)/5)
    test_examples = generate_examples(test_data, test_cores,1,1,1, len(test_cores)/5)
    print(len(train_examples))
    print("next")
    print(len(validation_examples))
    print("next")
    print(len(test_examples))
    count = 0
    count1 = 0
    for example in train_examples:
        if example[1] == "permits":
            count +=1
        if example[1] == "entails":
            count1 +=1
    print(count/len(train_examples))
    print(count1/len(train_examples))
    print(1 - (count + count1)/len(train_examples))
    train_examples += generate_boolean_examples(train_examples, len(train_examples))
    validation_examples += generate_boolean_examples(validation_examples, len(validation_examples))
    test_examples += generate_boolean_examples(test_examples, len(test_examples))
    print(len(train_examples))
    print("next")
    print(len(validation_examples))
    print("next")
    print(len(test_examples))
    count = 0
    count1 = 0
    for example in train_examples:
        if example[1] == "permits":
            count +=1
        if example[1] == "entails":
            count1 +=1
    print(count/len(train_examples))
    print(count1/len(train_examples))
    print(1 - (count + count1)/len(train_examples))
    random.shuffle(train_examples)
    save_data(train_examples[50000:], "train")
    save_data(validation_examples, "val")
    save_data(test_examples, "test")
    save_data(train_examples[0:50000], "jointtest")
