#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.

CODEBOOK=codebook_DBSCAN_cart_15000_12_1.3__0.3.csv

#parallel -N2 \
  ./vq.py -i ../../data/{1}.csv \
  -c ${CODEBOOK} \
  -o DBSCAN_cart_15000_12_1.3__0.3_${NPOI}_{2}_features.csv -N ${NPOI} \
  ::: training train   test test

# 554369.023743

./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/DBSCAN_cart_15000_12_1.3__0.3_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/DBSCAN_cart_15000_12_1.3__0.3_${NPOI}_train_features.csv \
  -o submission09.csv \
  -N 3000
