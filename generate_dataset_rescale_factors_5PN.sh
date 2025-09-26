dataset=${1:-"dataset_NS_5PN"}

#psd_path="npe/utils/psds/GW191216_213338_psd_H1.dat npe/utils/psds/GW191216_213338_psd_V1.dat"
psd_path="npe/utils/psds/GW170817_psd_H1.dat npe/utils/psds/GW170817_psd_L1.dat npe/utils/psds/GW170817_psd_V1.dat"
extra_args=""

#psd_path="npe/utils/asds/O3_H1_sensitivity.txt npe/utils/asds/O3_L1_sensitivity.txt npe/utils/asds/O3_V1_sensitivity.txt"
#extra_args=" --asd"

#min_args=" --min-over-tphi"
min_args=""

pool=4

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus1.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus2.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus3.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus4.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus5.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus6.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus7.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus8.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus9.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus10.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus11.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus12.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus13.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-minus0.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-plus1.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-plus2.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-plus3.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-plus4.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}

python npe/generate_dataset_phase_rescale_fac.py \
  --psd-path $psd_path \
  --dataset-path $dataset/ppe-plus5.pkl \
  --pool $pool \
  --output-rootdir ${dataset}_rescale \
  ${min_args}${extra_args}