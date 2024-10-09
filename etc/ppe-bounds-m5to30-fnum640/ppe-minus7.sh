eppe-create-dataset-ppe-bounds \
  --b-ppe -7 \
  --ppe-ref 10 \
  --ppe-ref-in-hz \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples 100000 \
  --seed 1234 \
  --pool 2 \
  --output-file ../../data/ppe-bounds-m5to30-fnum640/ppe-minus7.pkl \