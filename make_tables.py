import sys
import json

name = sys.argv[1]
crazy = sys.argv[2]
if bool(int(crazy)):
    name += "crazy"
prefixes = ["0", "0.0625", "0.125", "0.25", "0.375","0.5","0.625", "0.75"]
print("\\begin{tabular}{r| c| c| c| c| c| c |c |c}")
firstline = "&"
for p in prefixes:
    firstline += " " + p + " &"
print(firstline[:-2] + "\\\\")
print("\\hline")
data = dict()
for prefix in prefixes:
    with open(name + prefix + "gendataoutputstats") as f:
        data[prefix] = json.loads(f.readline())
trainline = "train &"
for p in prefixes:
    trainline += " $" + str(round(100*data[p][0], 2)) + "\\pm" + str(round(100*data[p][1], 2)) + "$ &"
print(trainline[:-2] + "\\\\")
print("\\hline")
devline = "dev &"
for p in prefixes:
    devline += " $" + str(round(100*data[p][2], 2)) + "\\pm" + str(round(100*data[p][3], 2)) + "$ &"
print(devline[:-2] + "\\\\")
print("\\hline")
testline = "test &"
for p in prefixes:
    testline += " $" + str(round(100*data[p][4],2)) + "\\pm" + str(round(100*data[p][5], 2)) + "$ &"
print(testline[:-2] + "\\\\")
print("\\end{tabular}")
