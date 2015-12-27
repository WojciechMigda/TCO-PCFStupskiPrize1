#!/bin/sh

PSZ=12
NPATCH=512
NPOIS=15000

cat ../../data/training.csv | ./cell_patches_kmeans.py \
  --patch-size=${PSZ} \
  --num-patches=${NPATCH} \
  --max-pois=${NPOIS} \
  -o cell_patches_kmeans_cart_${NPOIS}_${PSZ}_${NPATCH}.csv \

exit
cat ../../data/training.csv ../../data/test.csv | ./cell_patches_kmeans.py \
  --patch-size=${PSZ} \
  --num-patches=${NPATCH} \
  --max-pois=${NPOIS} \
  -o cell_patches_kmeans_cart_with_test_${NPOIS}_${PSZ}_${NPATCH}.csv \

