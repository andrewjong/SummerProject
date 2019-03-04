import sys

model= sys.argv[1]
attention= sys.argv[2]
crazy= sys.argv[3]
if bool(int(crazy)):
    name = "crazy"
else:
    name = ""
prefixes = ["0", "0.0625", "0.125", "0.25", "0.375","0.5","0.625", "0.75"]
print("export CONDA_HOME=/u/nlp/anaconda/ubuntu_16 \nexport PATH=${CONDA_HOME}/bin:/usr/local/cuda/bin:$PATH:/u/nlp/bin \nif [ ! $ANACONDA_ENV == '' ]; then \n    source activate $ANACONDA_ENV\nfi")
for prefix in prefixes:
    print("nlprun -g 1 -a atticus -o output" + model + attention + name + prefix + ".out " + "\'python train.py " + prefix + "gendata/"+ prefix + "gendata " +name + prefix + "gendata " + model + " " + attention + " 5 " + crazy + "\'")
