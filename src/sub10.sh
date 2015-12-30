#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.

CODEBOOK=codebook_kmeans_cart_15000_12_512__0.6.csv

parallel -N2 --dry-run \
  ./vq.py -i ../../data/{1}.csv \
  -c ${CODEBOOK} \
  -o KMeans_cart_15000_12_512__0.6_${NPOI}_{2}_features.csv -N ${NPOI} \
  ::: training train   test test

# 627655.471101

#exit
./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.6_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.6_${NPOI}_train_features.csv \
  -o submission10.csv \
  -N 3000
