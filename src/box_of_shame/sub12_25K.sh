#!/bin/sh -x

NPOI=25000
FEATDIR=../../data
FEATDIR=.
#DRYRUN=--dry-run

CODEBOOK=codebook_kmeans_cart_25000_12_512__0.4.csv

parallel -N2 ${DRYRUN} \
  ./vq.py -i ../../data/{1}.csv \
  -c ${CODEBOOK} \
  -o KMeans_cart_25000_12_512__0.4_${NPOI}_{2}_features.csv -N ${NPOI} \
  ::: training train   test test

# cv10 = 631058.259323 (1000)631029.535247
# cv33 = 616496.923267
# cv66 = 

exit
./clf_ert.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_test_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_${NPOI}_train_features.csv \
  -o submission11.csv \
  -N 3000
