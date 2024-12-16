systype=${1:-"BH"}
nsample=${2:-100000} # reproduces the paper
# nsample=1000 # for test runs

# TODO: Priority 1
# rescale frequency (f here is actually ~ fM, so NS ranges will be smaller --> find minimum based on LIGO sensitivity?)
# -----> try training VAE now and see what it looks like

# TODO: Priority 2
# find new way to rescale frequency bins (currently N=640, logarithmic spacing) ? --> training data dependent
# How to best include massive scalar theories (follow up with Nico on specifics)

if [[ "$systype" == "BH" ]]
then
  # Point Particle parameters
  M_MIN=5.0
  M_MAX=30.0
  CHI_MIN=-0.99
  CHI_MAX=0.099

  # Relevant frequency range
  F_MIN=0.0004
  F_MAX=0.018

elif [[ "$systype" == "NS" ]]
then
  # Point Particle parameters
  M_MIN=0.6
  M_MAX=3.0
  CHI_MIN=-0.05
  CHI_MAX=0.05

  # Tidal parameters (Unimplemented)
  C_MIN=0.1
  C_MAX=0.4
  L_MIN=0
  L_MAX=1

  # WIP: Relevant frequency range (double check these)
  #F_MIN=0.00003
  #F_MAX=0.025
  F_MIN=0.004 
  F_MAX=0.18 
else
  echo "Invalid systype: please select one of (BH, NS)"
  exit 1
fi

dataset_folder="dataset_$systype"

python npe/generate_dataset.py \
  --b-ppe -1 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus1.pkl

python npe/generate_dataset.py \
  --b-ppe -3 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus3.pkl

python npe/generate_dataset.py \
  --b-ppe -5 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus5.pkl

python npe/generate_dataset.py \
  --b-ppe -7 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus7.pkl

python npe/generate_dataset.py \
  --b-ppe -9 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus9.pkl

python npe/generate_dataset.py \
  --b-ppe -11 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus11.pkl

python npe/generate_dataset.py \
  --b-ppe -13 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --num-samples $nsample \
  --seed 1234 \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus13.pkl