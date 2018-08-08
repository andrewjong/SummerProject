import json
import data_util
import generate_data

def example_overlap(premise, hypothesis):
    count = 0
    if premise.subject_noun == hypothesis.subject_noun:
        count += 1
    if premise.object_noun == hypothesis.object_noun:
        count += 1
    if premise.verb == hypothesis.verb:
        count += 1
    if premise.subject_adjective== hypothesis.subject_adjective or hypothesis.subject_adjective =="" or premise.subject_adjective=="":
        count += 1
    if premise.object_adjective== hypothesis.object_adjective or hypothesis.object_adjective =="" or premise.object_adjective=="":
        count += 1
    if premise.adverb== hypothesis.adverb or hypothesis.adverb == "" or premise.adverb == "":
        count += 1
    return count

data, _, _ = generate_data.process_data(1.0)
labels = ["entails", "contradicts", "permits"]
experiment = "experiment1_level1"
with open(experiment + "results", "r") as f:
    training_data, predictions = json.loads(f.read())
examples = []
with open(experiment + ".val", "r") as f:
    for line in f:
        examples.append(json.loads(line))
mistakes = [0] * 7
successes = [0] * 7
totals = [0] * 7
for i in range(len(examples)):
    if examples[i]["gold_label"] == "permits":
        premise = data_util.parse_sentence(data, examples[i]["sentence1"])[0]
        hypothesis = data_util.parse_sentence(data,examples[i]["sentence2"])[0]
        totals[example_overlap(premise, hypothesis)] +=1
        if labels[predictions[i]] != examples[i]["gold_label"]:
            mistakes[example_overlap(premise, hypothesis)] +=1
        else:
            successes[example_overlap(premise, hypothesis)] +=1
for i in range(7):
    print(totals[i])
    print(mistakes[i]/totals[i], successes[i]/totals[i])
