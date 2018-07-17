import json
import random
num = int(input("How many examples would you like to see?"))
examples = []
with open("test", "r") as f:
    lines = f.readlines()
    for line in lines:
        examples.append(json.loads(line))
count = 0
for example in examples:
    if count == num:
        break
    print(example["sentence1"], "\n", example["gold_label"], "\n", example["sentence2"],"\n")
    count += 1
