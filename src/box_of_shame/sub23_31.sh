#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.
DRYRUN=--dry-run

CODEBOOK=codebook_kmeans_polar_15000_16_512__0.46.csv

parallel -N2 ${DRYRUN} \
  ./vq_polar.py -i ../../data/{1}.csv \
  -c ${CODEBOOK} \
  -o KMeans_polar_15000_16_512__0.46_${NPOI}_{2}_inv_features.csv \
  -N ${NPOI} \
  --inv-codebook \
  ::: training train   test test


# cv10 = 
# cv33 = 
# cv66 = 

#exit

# N=15 obtained with:
# for i in `seq 3 2 41`; do ./cross_val_knn.py --in-y-train-csv=../../data/training.csv --in-train-feat-csv=KMeans_cart_15000_12_512__0.4_15000_train_inv_features.csv  -k 131 -N ${i}  ; done

./clf_knn.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_polar_15000_16_512__0.46_${NPOI}_test_inv_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_polar_15000_16_512__0.46_${NPOI}_train_inv_features.csv \
  -o submission23_31.csv \
  -N 31 \
  -H
