## Info
This file provides some usage examples of HierCost
on small datasets.

for additional documentation on various CLI options
and file formats please visit https://cs.gmu.edu/~mlbio/HierCost/

HierCost examples are modified to suit Recursive Regularization

This code works for tree hierarchy (recursive regularization part)

## Usage
`cd rr-lr/`

for clef dataset:

`python src/train.py -d data/clef/train.txt -t data/clef/cat_hier.txt -m output/clef/model -f 80 -r 0.5 -c lr`

`python src/predict.py -m  output/clef/model -d  data/clef/test.txt -f 80 -p output/clef/pred.txt -t data/clef/cat_hier.txt`

For omniscience/swiki (rename address path):

`python src/train.py -d  data/oms/train_data.npy -t  data/oms/cat_hier_TREE_INT.graphml -m  output/oms/model -r 0.5 --multi -c lr`

`python src/predict.py --multi -m output/oms/model -d data/oms/test_data.npy -f 48732 -p output/oms/pred.txt -t data/oms/cat_hier_TREE_INT.graphml`





