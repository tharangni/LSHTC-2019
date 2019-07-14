MODELNAME=oms
MODELDIR=/home/harshasivajit/sspace/model/${MODELNAME}
DATADIR=/home/harshasivajit/sspace/data/${MODELNAME}
PREDDIR=/home/harshasivajit/sspace/pred/${MODELNAME}

echo "making directories"
mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"
mkdir -p "${PREDDIR}"


echo "Compiling StarSpace"

make

for dim in 300
do
    echo "Start to train on ${MODELNAME} data without hierarchy:"
    echo "dim ${dim}"
    /home/harshasivajit/StarSpace/starspace train \
    -trainFile "${DATADIR}/${MODELNAME}"-train.txt \
    -initModel "${MODELDIR}/${MODELNAME}-init-d${dim}".tsv \
    -model "${MODELDIR}/${MODELNAME}-d${dim}-neg-40-hless" \
    -adagrad true \
    -ngrams 1 \
    -lr 0.1 \
    -epoch 10 \
    -thread 50 \
    -dim ${dim} \
    -batchSize 25 \
    -negSearchLimit 40 \
    -trainMode 0 \
    -label "__label__" \
    -similarity "cosine" \
    -verbose true \
    -validationFile "${DATADIR}/${MODELNAME}-valid.txt" \
    -validationPatience 10 \

    echo "Start to evaluate trained model with dim ${dim}:"

    /home/harshasivajit/StarSpace/starspace test \
    -model "${MODELDIR}/${MODELNAME}-d${dim}-neg-40-hless" \
    -testFile "${DATADIR}/${MODELNAME}"-test.txt \
    -ngrams 1 \
    -dim ${dim} \
    -label "__label__" \
    -thread 50 \
    -batchSize 25 \
    -similarity "cosine" \
    -trainMode 0 \
    -verbose true \
    -adagrad true \
    -negSearchLimit 40 \
    -predictionFile "${PREDDIR}/${MODELNAME}-d${dim}-hless"-pred.txt \
    -K 5
    
done