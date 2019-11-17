# Project Varro
If FPGAs are universal function approximators, can they be used like neural networks?

This project expands upon Adrian Thompson's [famous paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.9691&rep=rep1&type=pdf) using modern FPGAs and advancements in evolutionary algorithms with the goal of universal function approximation (just like a neural network!)

## Installation

## Training (Fitting)
To train the individual to solve a problem, for example to evolve the neural network over 500 generations using the multiobjective novelty search - reward (nsr-es) just run:
> `python -m varro.algo.experiment --purpose 'fit' --cxpb 0 --ngen 500 --strategy 'nsr-es' --problem_type 'sinx'`

## Prediction
To predict from a checkpoint using a numpy file of inputs:
> `python -m varro.algo.experiment --purpose 'predict' --ckptfolder ./checkpoint/varro/algo/sinx_2019-Nov-16-19\:00\:41 --strategy 'nsr-es' --X ./varro/algo/X_test.npy`
