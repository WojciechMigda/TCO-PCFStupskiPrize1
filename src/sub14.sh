#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.
DRYRUN=--dry-run

CODEBOOK=codebook_kmeans_cart_15000_12_512__0.4.csv

parallel -N2 ${DRYRUN} \
  ./vq.py -i ../../data/{1}.csv \
  -c ${CODEBOOK} \
  -o KMeans_cart_15000_12_512__0.4_${NPOI}_{2}_features.csv -N ${NPOI} \
  ::: training train   test test


# cv10 = 
# cv33 = 
# cv66 = 

#exit
./clf_ada.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_train_features.csv \
  -o submission14.csv \
