#!/bin/bash -e
mkdir glove
wget http://nlp.stanford.edu/data/glove.6B.zip -P glove
unzip -j glove/glove.6B.zip glove.6B.300d.txt -d glove
rm glove/glove.6B.zip
echo "Extracting word list ..."
python -c "with open('glove/glove.6B.300d.txt','rb') as f: print('\n'.join([line.split()[0] for line in f]))" >> glove/glove.6B.list
echo "Done"
