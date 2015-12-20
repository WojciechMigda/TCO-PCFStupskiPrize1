#!/bin/sh

RADIUS=3
NPOI=50000
FEATDIR=../../data
#FEATDIR=.


parallel -N2 \
  ./lbp_features.py \
  -i ../../data/{1} \
  -o ${FEATDIR}/lbp24_np${NPOI}_r${RADIUS}_{2}.csv \
  -r ${RADIUS} \
  -N ${NPOI} \
  ::: training.csv train  test.csv test


./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/lbp24_np${NPOI}_r${RADIUS}_test.csv \
  --in-train-feat-csv ${FEATDIR}/lbp24_np${NPOI}_r${RADIUS}_train.csv \
  -o submission02.csv
