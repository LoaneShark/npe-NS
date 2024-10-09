nsample=100000 # reproduces the paper
# nsample=1000 # for test runs

python npe/generate_dataset.py \
  --b-ppe -1 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus1.pkl

python npe/generate_dataset.py \
  --b-ppe -3 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus3.pkl

python npe/generate_dataset.py \
  --b-ppe -5 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus5.pkl

python npe/generate_dataset.py \
  --b-ppe -7 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus7.pkl

python npe/generate_dataset.py \
  --b-ppe -9 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus9.pkl

python npe/generate_dataset.py \
  --b-ppe -11 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus11.pkl

python npe/generate_dataset.py \
  --b-ppe -13 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min 5.0 --m1-max 30.0 \
  --m2-min 5.0 --m2-max 30.0 \
  --chi1z-min -0.99 --chi1z-max 0.99 \
  --chi2z-min -0.99 --chi2z-max 0.99 \
  --fmin 0.0004 --fmax 0.018 \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file dataset/ppe-minus13.pkl