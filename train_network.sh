content=${1:-"BH"}

python npe/train_network.py \
  --dataset-rootdir dataset_$content \
  --output-rootdir network \
  --run-title npe-$content \
  --run-type $content