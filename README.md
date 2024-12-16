# Neural post-Einsteinian framework

This repo is for reproducing the results in the npE [paper](https://arxiv.org/abs/2403.18936).

## Short story

Run `generate_dataset.sh` to generate the dataset for training the npE networks.

Run `train_network.sh` to train the npE networks using the generated dataset. Also use [`tensorboard`](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) to monitor the training process.

The waveform model for GW parameter estimation is built based on the module `npe/waveform_analysis.py`.

## Code environment

For generating the dataset, you need `lal`, `astropy`, etc. 

For training the network, you need `pytorch` (cuda support is preferred but not necessary), `tensorboard`, etc.

For the waveform model, you need `bilby`, `lal`, etc.

A relatively simple solution that covers all is [`igwn-py310`](https://computing.docs.ligo.org/conda/environments/igwn-py310.html) plus `tensorboard`.
