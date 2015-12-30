#!/bin/sh -x

NPOI=15000
FEATDIR=../../data
FEATDIR=.
DRYRUN=--dry-run

CODEBOOK=codebook_kmeans_cart_15000_12_512__0.4.csv

# only training set is cloned
#./vq.py -i ../../data/training.csv \
#  -c ${CODEBOOK} \
#  -o KMeans_cart_15000_12_512__0.4_3x_${NPOI}_train_inv_features_x3.csv \
#  -N $(( 3 * ${NPOI} )) \
#  --inv-codebook \
#  -X 3

#./vq.py -i ../../data/test.csv \
#  -c ${CODEBOOK} \
#  -o KMeans_cart_15000_12_512__0.4_3x_${NPOI}_test_inv_features.csv \
#  -N $(( 3 * ${NPOI} )) \
#  --inv-codebook


# cv10 = 
# cv33 = 
# cv66 = 

#exit

# N=45 obtained with:
# for i in `seq 39 2 61`; do ./cross_val_knn.py --in-y-train-csv=../../data/training.csv --in-train-feat-csv=KMeans_cart_15000_12_512__0.4_45000_train_inv_features_x3.csv  -k 131 -N ${i} -X 3  ; done

./clf_knn.py \
  --in-y-train-csv ../../data/training.csv \
  --in-test-labels-csv ../../data/test.csv \
  --in-test-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_45000_test_inv_features.csv \
  --in-train-feat-csv ${FEATDIR}/KMeans_cart_15000_12_512__0.4_45000_train_inv_features_x3.csv \
  -o submission19.csv \
  -N 45 \
  -X 3
