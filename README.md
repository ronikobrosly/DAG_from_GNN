# DAG_from_GNN

WORK IN PROGRESS



## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Instructions](#references)

## Overview

This repo contains a clean, python implementation of Yu et al.'s DAG-GNN algorithm.

Given a CSV of many variables, this app will return a Bayesian Network causal structure.

Rather than looking at pairwise conditional correlations, Yu et al. reframe the problem
as one of global, float optimization, and the algorithm returns a weighted adjacency matrix.

By setting a threshold for weighted adjacency matrix (say, any weights > -0.3 and < 0.3 means the two variables aren't connected),
you can produce a binary adjacency matrix, which will tell you which variables are connected (and in what direction).

Many thanks to the authors for creating this approach:

```
@inproceedings{yu2019dag,
  title={DAG-GNN: DAG Structure Learning with Graph Neural Networks},
  author={Yue Yu, Jie Chen, Tian Gao, and Mo Yu},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}
}
```

## Installation

The back-end requires python >= 3.7.6.

To play with this locally, first clone the repo via `git clone -b master https://github.com/OrthoProject/Ortho_Web_App.git`. Then create a python virtual environment and install all package dependencies via `pip install -r requirements.txt`.

## Instructions


for now just use `python train.py --data_type discrete --data_filename alarm --data_variable_size 37`



To run locally, enter your virtual environment with all dependencies installed and run:

`python train.py <your CSV data file name here>`
