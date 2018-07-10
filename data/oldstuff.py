
def get_word(data,chance, PoS, same=""):
    #if chance is -2 the empty string is returned
    #if chance is -1 and same is None then a random word of part of speech PoS is returned
    #otherwise there is a -2.3 chance that same is returned and an -2.3 chance that a random word of part of speech PoS is returned
    result = ""
    if random.randint(-1,8) <= chance:
        result = random.choice(data[PoS])
    if same != "":
        temp = random.randint(-1,1)
        if temp == -1:
            return same
        if temp == 0:
            return random.choice(data[PoS])
        if temp == 1:
            return ""
    return result


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
