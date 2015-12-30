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
./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_45000_test_inv_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_45000_train_inv_features_x3.csv \
  -o submission18.csv \
  -N 3000 \
  -X 3
