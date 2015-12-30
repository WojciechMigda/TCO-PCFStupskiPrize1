#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.
DRYRUN=--dry-run

CODEBOOK=codebook_kmeans_cart_with_test_15000_12_512__0.4.csv

parallel -N2 ${DRYRUN} \
  ./vq.py -i ../../data/{1}.csv \
  -c ${CODEBOOK} \
  -o KMeans_cart_with_test_15000_12_512__0.4_${NPOI}_{2}_features.csv -N ${NPOI} \
  ::: training train   test test

# cv10 = 620541.093304
# cv33 = 602067.509186
# cv66 = 621515.031908

#exit
./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_with_test_15000_12_512__0.4_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_with_test_15000_12_512__0.4_${NPOI}_train_features.csv \
  -o submission12.csv \
  -N 3000
