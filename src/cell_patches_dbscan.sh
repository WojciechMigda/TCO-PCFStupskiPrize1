#!/bin/bash

NPOIS=15000
PSIZE=12
EPS=1.3

./cell_patches_dbscan.py \
  -i ../../data/training.csv \
  --patch-size=${PSIZE} \
  --max-pois=${NPOIS} \
  --epsilon=${EPS} \
#  -o cell_patches_dbscan_cart_${NPOIS}_${PSIZE}_${EPS}.csv \
#  --with-polar
