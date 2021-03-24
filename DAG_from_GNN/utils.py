"""
Utility functions
"""

import csv
import glob
import math
import os
import pickle
import re

import networkx as nx
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import scipy.linalg as slin
import scipy.sparse as sp
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.adam import Adam

from DAG_from_GNN.config import CONFIG


# ========================================
# VAE utility functions
# ========================================
def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise).double()
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def gauss_sample_z(logits, zsize):
    U = torch.randn(logits.size(0), zsize).double()
    x = torch.zeros(logits.size(0), 1, zsize).double()
    for j in range(logits.size(0)):
        x[j, 0, :] = (
            U[j, :] * torch.exp(logits[j, 0, zsize : 2 * zsize]) + logits[j, 0, 0:zsize]
        )
    return x


def gauss_sample_z_new(logits, zsize):
    U = torch.randn(logits.size(0), logits.size(1), zsize).double()
    x = torch.zeros(logits.size(0), logits.size(1), zsize).double()
    x[:, :, :] = U[:, :, :] + logits[:, :, 0:zsize]
    return x


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith("_graph" + extension))


def load_data(config, batch_size=1000, debug=False):

    # load dataset as numpy array
    X = np.expand_dims(
        np.loadtxt(
            os.path.expanduser("datasets/" + config.data_filename),
            skiprows=1,
            delimiter=",",
            dtype=np.int32,
        ),
        2,
    )

    # get column/variable names
    df_temp = pd.read_csv(os.path.expanduser("datasets/" + config.data_filename))
    config.column_names = df_temp.columns

    # get number of columns/variables
    config.data_variable_size = len(config.column_names)

    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, config


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.0
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.0
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = (
        data_norm
        + data_norm.transpose(2, 3)
        - 2 * torch.matmul(data, data.transpose(2, 3))
    )
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(
                np.all(np.vstack((dist > thresholds[i], dist <= thresholds[i + 1])), 0)
            )[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - torch.log(log_prior + eps))
    return kl_div.sum() / (num_atoms)


def kl_gaussian(preds, zsize):
    predsnew = preds.squeeze(1)
    mu = predsnew[:, 0:zsize]
    log_sigma = predsnew[:, zsize : 2 * zsize]
    kl_div = torch.exp(2 * log_sigma) - 2 * log_sigma + mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)) - zsize) * 0.5


def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5


def kl_categorical_uniform(
    preds, num_atoms, num_edge_types, add_const=False, eps=1e-16
):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_catogrical(preds, target, add_const=False):
    """compute the loglikelihood of discrete variables"""

    total_loss = 0
    for node_size in range(preds.size(1)):
        total_loss += -(
            torch.log(preds[:, node_size, target[:, node_size].long()])
            * target[:, node_size]
        ).mean()

    return total_loss


def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(
        torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance)
    )
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


# Symmetrically normalize adjacency matrix.
def normalize_adj(adj):
    rowsum = torch.abs(torch.sum(adj, 1))
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    myr = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    myr[isnan(myr)] = 0.0
    return myr


def preprocess_adj(adj):
    adj_normalized = torch.eye(adj.shape[0]).double() + (adj.transpose(0, 1))
    return adj_normalized


def preprocess_adj_new(adj):
    adj_normalized = torch.eye(adj.shape[0]).double() - (adj.transpose(0, 1))
    return adj_normalized


def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(
        torch.eye(adj.shape[0]).double() - adj.transpose(0, 1)
    )
    return adj_normalized


def isnan(x):
    return x != x


def my_normalize(z):
    znor = torch.zeros(z.size()).double()
    for i in range(z.size(0)):
        testnorm = torch.norm(z[i, :, :], dim=0)
        znor[i, :, :] = z[i, :, :] / testnorm
    znor[isnan(znor)] = 0.0
    return znor


def sparse_to_tuple(sparse_mx):
    #    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def matrix_poly(matrix, d):
    x = torch.eye(d).double() + torch.div(matrix, d)
    return torch.matrix_power(x, d)


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss += (
            2 * tol
            - torch.sum(torch.abs(A[:, i]))
            - torch.sum(torch.abs(A[i, :]))
            + z * z
        )
    return loss


# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):
    result = -A + z_positive * z_positive
    loss = torch.sum(result)

    return loss


"""
COMPUTE SCORES FOR BN
"""


def compute_BiCScore(G, D):
    """compute the bic score"""
    origin_score = []
    num_var = G.shape[0]
    for i in range(num_var):
        parents = np.where(G[:, i] != 0)
        score_one = compute_local_BiCScore(D, i, parents)
        origin_score.append(score_one)

    score = sum(origin_score)

    return score


def compute_local_BiCScore(np_data, target, parents):
    # use dictionary
    sample_size = np_data.shape[0]
    var_size = np_data.shape[1]

    # build dictionary and populate
    count_d = dict()
    if len(parents) < 1:
        a = 1

    for data_ind in range(sample_size):
        parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
        self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
        if parent_combination in count_d:
            if self_value in count_d[parent_combination]:
                count_d[parent_combination][self_value] += 1.0
            else:
                count_d[parent_combination][self_value] = 1.0
        else:
            count_d[parent_combination] = dict()
            count_d[parent_combination][self_value] = 1.0

    # compute likelihood
    loglik = 0.0

    num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)

    num_self_state = np.amax(np_data[:, target], axis=0) + 1

    for parents_state in count_d:
        local_count = sum(count_d[parents_state].values())
        for self_state in count_d[parents_state]:
            loglik += count_d[parents_state][self_state] * (
                math.log(count_d[parents_state][self_state] + 0.1)
                - math.log(local_count)
            )

    num_param = num_parent_state * (num_self_state - 1)
    bic = loglik - 0.5 * math.log(sample_size) * num_param

    return bic
