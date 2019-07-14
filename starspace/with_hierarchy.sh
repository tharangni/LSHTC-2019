
MODELNAME=oms
MODELDIR=/home/harshasivajit/sspace/model/${MODELNAME}
DATADIR=/home/harshasivajit/sspace/data/${MODELNAME}
PREDDIR=/home/harshasivajit/sspace/pred/${MODELNAME}
hneg=5
neg=40

echo "making directories"
mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"
mkdir -p "${PREDDIR}"


echo "Compiling StarSpace"

make

echo "Start to train on ${MODELNAME} data's hierarchy:"

for dim in 300
do
    echo "hierarchy hneg ${hneg}"
    /home/harshasivajit/StarSpace/starspace train \
    -trainFile "${DATADIR}/oms_2nd_dag2tree_fasttext_reverse".txt \
    -initModel "${MODELDIR}/${MODELNAME}-init-d${dim}".tsv \
    -model "${MODELDIR}/${MODELNAME}-d${dim}-h-only-neg-dag-${hneg}" \
    -adagrad true \
    -ngrams 1 \
    -lr 0.05 \
    -epoch 20 \
    -thread 50 \
    -dim ${dim} \
    -margin 0.05 \
    -batchSize 5 \
    -negSearchLimit ${hneg} \
    -maxNegSamples 10 \
    -trainMode 4 \
    -label "__label__" \
    -similarity "cosine" \
    -verbose true

    echo "Start to train on ${MODELNAME} data's documents with trained model on hierarchy:"

    echo "classification neg ${neg}"
    /home/harshasivajit/StarSpace/starspace train \
    -trainFile "${DATADIR}/${MODELNAME}"-train.txt \
    -initModel "${MODELDIR}/${MODELNAME}-d${dim}-h-only-neg-${hneg}".tsv \
    -model "${MODELDIR}/${MODELNAME}-d${dim}-neg-${neg}-h-${hneg}" \
    -adagrad true \
    -ngrams 1 \
    -lr 0.1 \
    -epoch 10 \
    -thread 50 \
    -dim ${dim} \
    -batchSize 25 \
    -negSearchLimit ${neg} \
    -trainMode 0 \
    -label "__label__" \
    -similarity "cosine" \
    -verbose true \
    -validationFile "${DATADIR}/${MODELNAME}-valid.txt" \
    -validationPatience 10 \

    echo "Start to evaluate trained model with clf neg ${neg} and h-neg ${hneg}:"

    /home/harshasivajit/StarSpace/starspace test \
    -model "${MODELDIR}/${MODELNAME}-d${dim}-neg-${neg}-h-${hneg}" \
    -testFile "${DATADIR}/${MODELNAME}"-test.txt \
    -ngrams 1 \
    -dim ${dim} \
    -label "__label__" \
    -thread 50 \
    -batchSize 10 \
    -similarity "cosine" \
    -trainMode 0 \
    -verbose true \
    -adagrad true \
    -negSearchLimit ${neg} \
    -predictionFile "${PREDDIR}/${MODELNAME}-d${dim}-neg-${neg}-h-${hneg}"-pred.txt \
    -K 5
done