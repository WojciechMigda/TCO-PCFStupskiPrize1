# TCO-PCFStupskiPrize1
Pipeline outline:
 1. preprocess images (trimming, alien stain cleaning, ...)
 2. colorspace conversion to HED and leaving only the Hematoxin layer
 3. histogram equalization
 4. detection of points of interest (POIs)
 5. initial KMeans clustering of patches around POIs,
 6. secondary clustering using DBSCAN to yield codebook of cellular patches
 7. generation of histograms from quantized images
 8. prediction using KNN model with Hellinger distance metric over data with the most dominating feature removed.

Commands which yield the highest-scored submission (#24):

 1. Initial clustering

 `cat ../../data/training.csv | ./cell_patches_kmeans.py --patch-size=12 --num-patches=512 --max-pois=15000 -o cell_patches_kmeans_cart_15000_12_512.csv`
 2. Codebook generation

 `./codebook.py -i cell_patches_kmeans_cart_15000_12_512.csv -o codebook_kmeans_cart_15000_12_512__0.4.csv -e 0.4`

 3. Quantization into train and test histograms

 `parallel -N2 ./vq.py -i ../../data/{1}.csv 
 -c codebook_kmeans_cart_15000_12_512__0.4.csv 
 -o KMeans_cart_15000_12_512__0.4_15000_{2}_inv_features.csv 
 -N 15000 --inv-codebook ::: training train   test test`

 4. Training & prediction

 (from `sub24.sh`)

 `./clf_knn.py 
  --in-y-train-csv ../../data/training.csv 
  --in-test-labels-csv ../../data/test.csv 
  --in-test-feat-csv KMeans_cart_15000_12_512__0.4_15000_test_inv_features.csv 
  --in-train-feat-csv KMeans_cart_15000_12_512__0.4_15000_train_inv_features.csv 
  -o submission24.csv -N 15 -H -R 105`
