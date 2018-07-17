import generate_data as gd
import fol_model as fol
import natural_logic_model as nlm
import util
import json

data, _, _ = gd.process_data(1.0)
with open("simple_solutions","r") as f:
    simple_solutions = json.loads(f.read())
for encoding in simple_solutions:
    encoding = json.loads(encoding)
    premise, hypothesis = gd.encoding_to_example(data,encoding)
    nlm_label = nlm.get_label(nlm.compute_simple_relation(premise, hypothesis))
    if simple_solutions[json.dumps(encoding)] != nlm_label:
        print("We have a problem with the simple file")
print("simple file is good")
with open("boolean_solutions","r") as f:
    boolean_solutions = json.loads(f.read())
simple_entails1 = (util.parse_sentence(data, "every wizard eats every flute")[0],util.parse_sentence(data, "some wizard eats some flute")[0])
simple_contradicts1 = (util.parse_sentence(data, "no wizard eats some flute")[0],util.parse_sentence(data, "some wizard eats every flute")[0])
simple_permits1 = (util.parse_sentence(data, "no wizard eats some tree")[0],util.parse_sentence(data, "some wizard eats every flute")[0])
simple_entails2 = (util.parse_sentence(data, "every wizard eats every vape")[0],util.parse_sentence(data, "some wizard eats some vape")[0])
simple_contradicts2 = (util.parse_sentence(data, "no wizard eats some vape")[0],util.parse_sentence(data, "some wizard eats every vape")[0])
simple_permits2 = (util.parse_sentence(data, "no wizard eats some tree")[0],util.parse_sentence(data, "some wizard eats every vape")[0])
for encoding in boolean_solutions:
    encoding = json.loads(encoding)
    conjunctions = ["or", "and", "then"]
    premise_conjunction = conjunctions[encoding[0]]
    hypothesis_conjunction = conjunctions[encoding[1]]
    if encoding[2]== 0:
        premise1, hypothesis1 = simple_entails1
    if encoding[2]== 1:
        premise1, hypothesis1 = simple_contradicts1
    if encoding[2]== 2:
        premise1, hypothesis1 = simple_permits1
    if encoding[3]== 0:
        premise2, hypothesis2 = simple_entails2
    if encoding[3]== 1:
        premise2, hypothesis2 = simple_contradicts2
    if encoding[3]== 2:
        premise2, hypothesis2 = simple_permits2
    nlm_label = nlm.get_label(nlm.compute_boolean_relation(premise1, premise_conjunction, premise2, hypothesis1, hypothesis_conjunction, hypothesis2))
    if boolean_solutions[json.dumps(encoding)] != nlm_label:
        print("We have a problem with the boolean file")
print("boolean file is good")
examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 1000, 1000, data)
gd.save_data(examples, "test")
examples = []
with open("test", "r") as f:
    lines = f.readlines()
    for line in lines:
        examples.append(json.loads(line))
for example in examples:
    premise = util.parse_sentence(data,example["sentence1"])
    hypothesis = util.parse_sentence(data,example["sentence2"])
    if len(premise) == 1:
        fol_label = fol.get_label(premise[0], hypothesis[0])
        nlm_label = nlm.get_label(nlm.compute_simple_relation(premise[0], hypothesis[0]))
        if example["gold_label"] != fol_label or fol_label != nlm_label:
            print("We have a problem with simple generation")
    else:
        premise1 = premise[0]
        premise_conjunction = premise[1]
        premise2 = premise[2]
        hypothesis1 = hypothesis[0]
        hypothesis_conjunction = hypothesis[1]
        hypothesis2 = hypothesis[2]
        nlm_label = nlm.get_label(nlm.compute_boolean_relation(premise1, premise_conjunction, premise2, hypothesis1, hypothesis_conjunction, hypothesis2))
        if example["gold_label"] != nlm_label:
            print("We have a problem with boolean generation")
print("generation is good")
