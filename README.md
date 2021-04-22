# DAG_from_GNN (Directed Acyclic Graphs from Graph Neural Networks)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Instructions](#references)

## Overview

This repo contains a clean, python implementation of Yu et al.'s DAG-GNN algorithm.

Given a CSV of many variables, this app will learn the structure of a Bayesian Belief Network.

Rather than looking at pairwise conditional correlations, Yu et al. reframe the problem
as one of optimization of a continuous function, and the algorithm returns a weighted adjacency matrix.

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

To play with this locally, first clone the repo via `git clone -b main git@github.com:ronikobrosly/DAG_from_GNN.git`. Then create a python virtual environment and install all package dependencies via `pip install -r requirements.txt`.

## Instructions

To run locally, enter your virtual environment and ensure that all dependencies have been installed.
Ensure that your input data is a properly formatted CSV file. The first line of the file should
be the header with the column names. Cell values below the header should only be integers,
representing categories (ensure these are only categorical variables). Place this input CSV
in the `datasets` folder of the app. There are two example datasets already in the folder.

Open `DAG_from_GNN/config.py` in your editor and edit parameters as you see fit.
Note: At a minimum, you must change the `data_filename` parameter to match the filename
of your input dataset.

Now run the following:

`python -m DAG_from_GNN`

Press `CTRL + C` to stop training. Once training has completed, results will be
saved in the `results` folder. These include:

* A plot of the final estimated structure of the DAG as a `.png` file
* A CSV of the final estimated adjacency matrix
