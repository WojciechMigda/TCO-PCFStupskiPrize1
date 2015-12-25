#!/bin/sh

NPOI=25000
FEATDIR=../../data
FEATDIR=.

./vq.py -i ../../data/training.csv -o DBSCAN_cart_${NPOI}_train_features.csv -N ${NPOI}

./vq.py -i ../../data/test.csv -o DBSCAN_cart_${NPOI}_test_features.csv -N ${NPOI}

./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/DBSCAN_cart_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/DBSCAN_cart_${NPOI}_train_features.csv \
  -o submission06.csv
