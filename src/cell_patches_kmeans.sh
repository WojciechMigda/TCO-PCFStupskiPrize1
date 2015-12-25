#!/bin/sh

./cell_patches_kmeans.py \
  -i ../../data/training.csv \
  -o cell_patches_kmeans.csv \
  --patch-size=16 \
  --num-patches=80 \
  --max-pois=5000
