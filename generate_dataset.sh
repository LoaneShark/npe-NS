systype=${1:-"BH"}
nsample=${2:-100000} # reproduces the paper
# nsample=1000 # for test runs
seed=${3:-1234}
include_halfPN=${4:-0}
include_tidal=${5:-0}
include_5PN=${6:-0}
include_2PN=${7:-1}


# Black Hole parameters
M_MIN_BH=5.0
M_MAX_BH=30.0
CHI_MIN_BH=-0.99
CHI_MAX_BH=0.99

# Neutron Star parameters
M_MIN_NS=0.5
M_MAX_NS=3.0
CHI_MIN_NS=-0.1
CHI_MAX_NS=0.1

# Dimensionless tidal parameters
L_MIN=0
L_MAX=5000

# Quadrupole-monopole parameters
# NOTE: Set to 0 because these are currently inferred from Lambdas with Love-Q relations
CQ_MIN=0
CQ_MAX=0
#CQ_MIN=1
#CQ_MAX=30

# Compactness and radius (unused)
#C_MIN_NS=0.1
#C_MAX_NS=0.4
#R_MIN_NS=9
#R_MAX_NS=12


if [[ "$systype" == "BH" ]]
then
  # Point Particle BH parameters
  M_MIN_1=$M_MIN_BH
  M_MIN_2=$M_MIN_BH
  M_MAX_1=$M_MAX_BH
  M_MAX_2=$M_MAX_BH
  CHI_MIN_1=$CHI_MIN_BH
  CHI_MIN_2=$CHI_MIN_BH
  CHI_MAX_1=$CHI_MAX_BH
  CHI_MAX_2=$CHI_MAX_BH

  # Dimensionless BH tidal parameters (set to zero)
  L_MIN_1=0
  L_MIN_2=0
  L_MAX_1=0
  L_MAX_2=0
  CQ_MIN_1=0
  CQ_MIN_2=0
  CQ_MAX_1=0
  CQ_MAX_2=0

  # Relevant frequency range
  F_MIN=0.0004    # 10 Hz (detector limit)
  F_MAX=0.018     # IMRPhenomD inspiral cutoff
  F_NUM=640       # Number of frequency points

  TIDAL_TERMS=""
  EXTRA_TERMS=""

elif [[ "$systype" == "NS" ]]
then
  # Point Particle NS parameters
  M_MIN_1=$M_MIN_NS
  M_MIN_2=$M_MIN_NS
  M_MAX_1=$M_MAX_NS
  M_MAX_2=$M_MAX_NS
  CHI_MIN_1=$CHI_MIN_NS
  CHI_MIN_2=$CHI_MIN_NS
  CHI_MAX_1=$CHI_MAX_NS
  CHI_MAX_2=$CHI_MAX_NS

  # Dimensionless NS tidal parameters
  L_MIN_1=$L_MIN
  L_MIN_2=$L_MIN
  L_MAX_1=$L_MAX
  L_MAX_2=$L_MAX
  CQ_MIN_1=$CQ_MIN
  CQ_MIN_2=$CQ_MIN
  CQ_MAX_1=$CQ_MAX
  CQ_MAX_2=$CQ_MAX

  # Compactness and radius (unused)
  #C_MIN_1=$C_MIN_NS
  #C_MIN_2=$C_MIN_NS
  #C_MAX_1=$C_MAX_NS
  #C_MAX_2=$C_MAX_NS
  #R_MIN_1=$R_MIN_NS
  #R_MIN_2=$R_MIN_NS
  #R_MAX_1=$R_MAX_NS
  #R_MAX_2=$R_MAX_NS

  # Relevant frequency range
  F_MIN=0.00004   # 10 Hz (detector limit)
  F_MAX=0.018     # IMRPhenomD inspiral cutoff
  F_NUM=640       # Number of frequency points
  #F_NUM=1280      # Number of frequency points

  if [ "$include_tidal" -eq "0" ]; then
    TIDAL_TERMS=""
  else
    TIDAL_TERMS="--include-tidal"
    #TIDAL_TERMS="--include-tidal-full"
    echo "Including tidal parameters in dataset labels"
  fi

  #EXTRA_TERMS="--freqs-using-mc"
  EXTRA_TERMS=""

elif [[ "$systype" == "NSBH" ]]
then
  # Point Particle NS parameters
  # TODO: Should we limit the BH mass range for NSBH?
  # For now, we use the same BH parameters as in the BH case.
  M_MIN_1=$M_MIN_BH
  M_MIN_2=$M_MIN_NS
  M_MAX_1=$M_MAX_BH
  M_MAX_2=$M_MAX_NS
  CHI_MIN_1=$CHI_MIN_BH
  CHI_MIN_2=$CHI_MIN_NS
  CHI_MAX_1=$CHI_MAX_BH
  CHI_MAX_2=$CHI_MAX_NS

  # Dimensionless NS tidal parameters
  L_MIN_1=0
  L_MIN_2=$L_MIN
  L_MAX_1=0
  L_MAX_2=$L_MAX
  CQ_MIN_1=0
  CQ_MIN_2=$CQ_MIN
  CQ_MAX_1=0
  CQ_MAX_2=$CQ_MAX

  # Compactness and radius (unused)
  #C_MIN_1=1
  #C_MIN_2=$C_MIN_NS
  #C_MAX_1=1
  #C_MAX_2=$C_MAX_NS
  #R_MIN_1=0
  #R_MIN_2=$R_MIN_NS
  #R_MAX_1=0
  #R_MAX_2=$R_MAX_NS

  # Relevant frequency range
  # TODO: Determine what the f_min and f_max should be for NSBH
  F_MIN=0.00004   # 10 Hz (detector limit)
  F_MAX=0.018     # IMRPhenomD inspiral cutoff
  F_NUM=640       # Number of frequency points

  if [ "$include_tidal" -eq "0" ]; then
    TIDAL_TERMS=""
  else
    TIDAL_TERMS="--include-tidal"
    #TIDAL_TERMS="--include-tidal-full"
    echo "Including tidal parameters in dataset labels"
  fi

  EXTRA_TERMS=""

elif [[ "$systype" == "CBC" ]]
then
  # Point Particle CBC parameters
  # TODO: Should we enforce any sort of similarity in the two object masses?
  # For now, allow mass range to be uniformly distributed across the entire NS and BH mass range.
  M_MIN_1=$M_MIN_NS
  M_MIN_2=$M_MIN_NS
  M_MAX_1=$M_MAX_BH
  M_MAX_2=$M_MAX_BH
  # Use BH spin limits for both objects
  CHI_MIN_1=$CHI_MIN_BH
  CHI_MIN_2=$CHI_MIN_BH
  CHI_MAX_1=$CHI_MAX_BH
  CHI_MAX_2=$CHI_MAX_BH

  # Use NS tidal limits for both objects
  L_MIN_1=$L_MIN
  L_MIN_2=$L_MIN
  L_MAX_1=$L_MAX
  L_MAX_2=$L_MAX
  CQ_MIN_1=$CQ_MIN
  CQ_MIN_2=$CQ_MIN
  CQ_MAX_1=$CQ_MAX
  CQ_MAX_2=$CQ_MAX

  # Compactness and radius (unused)
  #C_MIN_1=$C_MIN_NS
  #C_MIN_2=$C_MIN_NS
  #C_MAX_1=$C_MAX_NS
  #C_MAX_2=$C_MAX_NS
  #R_MIN_1=$R_MIN_NS
  #R_MIN_2=$R_MIN_NS
  #R_MAX_1=$R_MAX_NS
  #R_MAX_2=$R_MAX_NS

  # Relevant frequency range
  # TODO: Determine what the f_min and f_max should be for this arbitrary CBC case
  F_MIN=0.00004   # 10 Hz (detector limit)
  F_MAX=0.018     # IMRPhenomD inspiral cutoff
  F_NUM=640       # Number of frequency points

  if [ "$include_tidal" -eq "0" ]; then
    TIDAL_TERMS=""
  else
    TIDAL_TERMS="--include-tidal"
    #TIDAL_TERMS="--include-tidal-full"
    echo "Including tidal parameters in dataset labels"
  fi

  EXTRA_TERMS=""

else
  echo "Invalid systype: please select one of (BH, NS, NSBH, CBC)"
  exit 1
fi

dataset_folder="dataset_${systype}"

if [ "$include_2PN" -eq "1" ]; then
  python npe/generate_dataset.py \
    --b-ppe -1 \
    --n-ppe 1 \
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
    --logspace-freqs \
    --freq-in-geometric-units \
    --ppe-ref-min 10 \
    --num-samples $nsample \
    --seed $seed \
    --pool 2 \
    --output-file $dataset_folder/ppe-minus1.pkl
fi
  
python npe/generate_dataset.py \
  --b-ppe -2 \
  --n-ppe 1 \
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
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
  --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
  --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
  --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
  --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
  --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
  --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
  --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
  --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
  --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
  --fmin $F_MIN --fmax $F_MAX \
  --num-freqs $F_NUM \
  --logspace-freqs \
  --freq-in-geometric-units \
  --ppe-ref-min 10 \
  --num-samples $nsample \
  --seed $seed \
  --pool 2 \
  --output-file $dataset_folder/ppe-minus13.pkl


if [ "$include_5PN" -eq "0" ]; then
  exit
else
  python npe/generate_dataset.py \
    --b-ppe 0 \
    --n-ppe 1 \
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
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
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
    --logspace-freqs \
    --freq-in-geometric-units \
    --ppe-ref-min 10 \
    --num-samples $nsample \
    --seed $seed \
    --pool 2 \
    --output-file $dataset_folder/ppe-plus1.pkl

  python npe/generate_dataset.py \
    --b-ppe 2 \
    --n-ppe 1 \
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
    --logspace-freqs \
    --freq-in-geometric-units \
    --ppe-ref-min 10 \
    --num-samples $nsample \
    --seed $seed \
    --pool 2 \
    --output-file $dataset_folder/ppe-plus2.pkl

  python npe/generate_dataset.py \
    --b-ppe 3 \
    --n-ppe 1 \
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
    --logspace-freqs \
    --freq-in-geometric-units \
    --ppe-ref-min 10 \
    --num-samples $nsample \
    --seed $seed \
    --pool 2 \
    --output-file $dataset_folder/ppe-plus3.pkl

  python npe/generate_dataset.py \
    --b-ppe 4 \
    --n-ppe 1 \
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
    --logspace-freqs \
    --freq-in-geometric-units \
    --ppe-ref-min 10 \
    --num-samples $nsample \
    --seed $seed \
    --pool 2 \
    --output-file $dataset_folder/ppe-plus4.pkl

  python npe/generate_dataset.py \
    --b-ppe 5 \
    --n-ppe 1 \
    --minus-gr $TIDAL_TERMS $EXTRA_TERMS \
    --m1-min $M_MIN_1 --m1-max $M_MAX_1 \
    --m2-min $M_MIN_2 --m2-max $M_MAX_2 \
    --chi1z-min $CHI_MIN_1 --chi1z-max $CHI_MAX_1 \
    --chi2z-min $CHI_MIN_2 --chi2z-max $CHI_MAX_2 \
    --l1-min $L_MIN_1 --l1-max $L_MAX_1 \
    --l2-min $L_MIN_2 --l2-max $L_MAX_2 \
    --cq1-min $CQ_MIN_1 --cq1-max $CQ_MAX_1 \
    --cq2-min $CQ_MIN_2 --cq2-max $CQ_MAX_2 \
    --fmin $F_MIN --fmax $F_MAX \
    --num-freqs $F_NUM \
    --logspace-freqs \
    --freq-in-geometric-units \
    --ppe-ref-min 10 \
    --num-samples $nsample \
    --seed $seed \
    --pool 2 \
    --output-file $dataset_folder/ppe-plus5.pkl
fi