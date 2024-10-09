eppe-create-dataset-ppe \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.9 --chi1z-max 0.9 \
  --chi2z-min -0.9 --chi2z-max 0.9 \
  --b-ppe -7 \
  --ppe-saturate-perturbation \
  --labels-only \
  --num-samples 100000 \
  --seed 1234 \
  --pool 4 \
  --output-file ../../data/ppe-minus5-labels-only-saturate-perturb-m5to30.pkl \