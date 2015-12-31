#!/bin/sh

NPOI=5000
FEATDIR=../../data
FEATDIR=.

./vq.py -i ../../data/training.csv -c codebook_DBSCAN_polar_5000_14_1.6__0.4.csv -o DBSCAN_polar_5000_14_1.6__0.4_${NPOI}_train_features.csv -N ${NPOI}

./vq.py -i ../../data/test.csv -c codebook_DBSCAN_polar_5000_14_1.6__0.4.csv -o DBSCAN_polar_5000_14_1.6__0.4_${NPOI}_test_features.csv -N ${NPOI}

# not submitted - CV = 442599

#./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/DBSCAN_cart_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/DBSCAN_cart_${NPOI}_train_features.csv \
  -o submission06.csv \
  -N 3000
