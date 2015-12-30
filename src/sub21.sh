#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.

./clf_xgb.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_test_inv_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_train_inv_features.csv \
  -o submission21.csv \
  --n-estimators=1000 \
  --seed=1 \
  --max-depth=3
