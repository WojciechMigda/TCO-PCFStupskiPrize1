#!/bin/sh

RADIUS=3
NPOI=20000
FEATDIR=../../data
FEATDIR=.

./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/DBSCAN_cart_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/DBSCAN_cart_train_features.csv \
  -o submission05.csv
