systype=${1:-"BH"}
nsample=${2:-100000} # reproduces the paper
# nsample=1000 # for test runs
seed=${3:-1234}

# TODO: Priority 1
# rescale frequency (f here is actually ~ fM, so NS ranges will be smaller --> find minimum based on LIGO sensitivity?)
# -----> try training VAE now and see what it looks like

# TODO: Priority 2
# find new way to rescale frequency bins (currently N=640, logarithmic spacing) ? --> training data dependent
# How to best include massive scalar theories (follow up with Nico on specifics)

if [[ "$systype" == "BH" ]]
then
  # Point Particle BH parameters
  M_MIN=5.0
  M_MAX=30.0
  CHI_MIN=-0.99
  CHI_MAX=0.99

  # Dimensionless BH tidal parameters
  L_MIN=0
  L_MAX=0
  CQ_MIN=0
  CQ_MAX=0
  #CQ_MIN=1
  #CQ_MAX=1

  # Relevant frequency range
  F_MIN=0.0004    # 10 Hz (detector limit)
  F_MAX=0.018     # IMRPhenomD inspiral cutoff

elif [[ "$systype" == "NS" ]]
then
  # Point Particle NS parameters
  M_MIN=0.6
  M_MAX=3.0
  CHI_MIN=-0.1
  CHI_MAX=0.1

  # Dimensionless NS tidal parameters
  L_MIN=0
  #L_MAX=5000
  L_MAX=0
  CQ_MIN=0
  CQ_MAX=0
  #CQ_MIN=1
  #CQ_MAX=10

  # Compactness and radius (unused)
  #C_MIN=0.1
  #C_MAX=0.4
  #R_MIN=9
  #R_MAX=12

  # Relevant frequency range
  # TODO: Double check f_min and f_max for IMRPhenomPv2_NRTidalv2 case
  F_MIN=0.00004   # 10 Hz (detector limit)
  F_MAX=0.018     # IMRPhenomD inspiral cutoff
else
  echo "Invalid systype: please select one of (BH, NS)"
  exit 1
fi

dataset_folder="dataset_${systype}"

python npe/generate_dataset.py \
  --b-ppe -1 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus1.pkl
  
python npe/generate_dataset.py \
  --b-ppe -2 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus2.pkl

python npe/generate_dataset.py \
  --b-ppe -3 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus3.pkl

python npe/generate_dataset.py \
  --b-ppe -4 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus4.pkl

python npe/generate_dataset.py \
  --b-ppe -5 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus5.pkl


python npe/generate_dataset.py \
  --b-ppe -6 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus6.pkl

python npe/generate_dataset.py \
  --b-ppe -7 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus7.pkl

python npe/generate_dataset.py \
  --b-ppe -8 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus8.pkl

python npe/generate_dataset.py \
  --b-ppe -9 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus9.pkl

python npe/generate_dataset.py \
  --b-ppe -10 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus10.pkl

python npe/generate_dataset.py \
  --b-ppe -11 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus11.pkl

python npe/generate_dataset.py \
  --b-ppe -12 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus12.pkl

python npe/generate_dataset.py \
  --b-ppe -13 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus13.pkl

## DEBUG DATASETS
'''
python npe/generate_dataset.py \
  --b-ppe 0 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus0.pkl

python npe/generate_dataset.py \
  --b-ppe 1 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-plus1.pkl

python npe/generate_dataset.py \
  --b-ppe -14 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus14.pkl

python npe/generate_dataset.py \
  --b-ppe -15 \
  --n-ppe 1 \
  --minus-gr \
  --m1-min $M_MIN --m1-max $M_MAX \
  --m2-min $M_MIN --m2-max $M_MAX \
  --chi1z-min $CHI_MIN --chi1z-max $CHI_MAX \
  --chi2z-min $CHI_MIN --chi2z-max $CHI_MAX \
  --l1-min $L_MIN --l1-max $L_MAX \
  --l2-min $L_MIN --l2-max $L_MAX \
  --cq1-min $CQ_MIN --cq1-max $CQ_MAX \
  --cq2-min $CQ_MIN --cq2-max $CQ_MAX \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs 640 \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus15.pkl
'''