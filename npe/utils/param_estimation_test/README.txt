npE parameter estimation test

Injection model: ppE
Recovery model: npE
This should be able to reproduce Fig.7 - Fig.9 in the paper

Files:
npe_wf_analysis.py - npe model, construction of the npe dephasing
bilby_script.py - main script for injection and recovery
submit.sh - bash script for submiting the job to slurm

File not included:
npe_network.pt - torch file saving the npe network. 
You can get a copy from `/projects/illinois/eng/physics/nyunes/yiqixie2/npe_realdata/npe_network.pt`

Note:
To run bilby_script.py, you need to specify the ppE b and beta for the injection
The beta value is taken relative to the post-Einsteinian boundary
The example in submit.sh should reproduce Fig.8(a)