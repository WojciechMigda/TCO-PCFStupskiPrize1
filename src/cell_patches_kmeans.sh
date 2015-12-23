#!/bin/sh

./cell_patches.py \
  -i ../../data/training.csv \
  -o cell_patches.csv \
  --patch-size=16 \
  --num-patches=80 \
  --max-pois=5000
