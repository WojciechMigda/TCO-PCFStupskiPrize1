#!/bin/sh

NPOI=5000
FEATDIR=../../data
FEATDIR=.

./vq.py -i ../../data/training.csv -c codebook_DBSCAN_polar_0.25.csv -o ${FEATDIR}/DBSCAN_polar_${NPOI}_train_features.csv -N ${NPOI}

./vq.py -i ../../data/test.csv -c codebook_DBSCAN_polar_0.25.csv -o ${FEATDIR}/DBSCAN_polar_${NPOI}_test_features.csv -N ${NPOI}

./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/DBSCAN_polar_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/DBSCAN_polar_${NPOI}_train_features.csv \
  -o submission08.csv
