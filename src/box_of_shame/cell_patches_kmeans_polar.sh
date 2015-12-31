#!/bin/sh

PSZ=18
NPATCH=512
NPOIS=15000

cat ../../data/training.csv | ./cell_patches_kmeans.py \
  --patch-size=${PSZ} \
  --num-patches=${NPATCH} \
  --max-pois=${NPOIS} \
  -o cell_patches_kmeans_polar_${NPOIS}_${PSZ}_${NPATCH}.csv \
  --with-polar
