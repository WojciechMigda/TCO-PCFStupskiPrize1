#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.

./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_test_inv_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_train_inv_features.csv \
  -o submission22.csv \
  -N 3000
