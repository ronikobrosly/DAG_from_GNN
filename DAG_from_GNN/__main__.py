"""
Main function for traininng DAG-GNN
"""

import argparse
import datetime
import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler

from DAG_from_GNN.utils import *
from DAG_from_GNN.modules import *
from DAG_from_GNN.config import CONFIG


CONFIG.cuda = not CONFIG.no_cuda and torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor


torch.manual_seed(CONFIG.seed)
if CONFIG.cuda:
    torch.cuda.manual_seed(CONFIG.seed)


# ================================================
# get data: experiments = {synthetic SEM, ALARM}
# ================================================
# train_loader, valid_loader, test_loader, ground_truth_G = load_data( args, CONFIG.batch_size, CONFIG.suffix)
train_loader, valid_loader, test_loader, CONFIG = load_data(CONFIG, CONFIG.batch_size)


# ===================================
# load modules
# ===================================
# Generate off-diagonal interaction graph
off_diag = np.ones([CONFIG.data_variable_size, CONFIG.data_variable_size]) - np.eye(
    CONFIG.data_variable_size
)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

# add adjacency matrix A
num_nodes = CONFIG.data_variable_size
adj_A = np.zeros((num_nodes, num_nodes))


if CONFIG.encoder == "mlp":
    encoder = MLPEncoder(
        CONFIG.data_variable_size * CONFIG.x_dims,
        CONFIG.x_dims,
        CONFIG.encoder_hidden,
        int(CONFIG.z_dims),
        adj_A,
        batch_size=CONFIG.batch_size,
        do_prob=CONFIG.encoder_dropout,
        factor=CONFIG.factor,
    ).double()
elif CONFIG.encoder == "sem":
    encoder = SEMEncoder(
        CONFIG.data_variable_size * CONFIG.x_dims,
        CONFIG.encoder_hidden,
        int(CONFIG.z_dims),
        adj_A,
        batch_size=CONFIG.batch_size,
        do_prob=CONFIG.encoder_dropout,
        factor=CONFIG.factor,
    ).double()

if CONFIG.decoder == "mlp":
    decoder = MLPDecoder(
        CONFIG.data_variable_size * CONFIG.x_dims,
        CONFIG.z_dims,
        CONFIG.x_dims,
        encoder,
        data_variable_size=CONFIG.data_variable_size,
        batch_size=CONFIG.batch_size,
        n_hid=CONFIG.decoder_hidden,
        do_prob=CONFIG.decoder_dropout,
    ).double()
elif CONFIG.decoder == "sem":
    decoder = SEMDecoder(
        CONFIG.data_variable_size * CONFIG.x_dims,
        CONFIG.z_dims,
        2,
        encoder,
        data_variable_size=CONFIG.data_variable_size,
        batch_size=CONFIG.batch_size,
        n_hid=CONFIG.decoder_hidden,
        do_prob=CONFIG.decoder_dropout,
    ).double()


# ===================================
# set up training parameters
# ===================================
if CONFIG.optimizer == "Adam":
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr
    )
elif CONFIG.optimizer == "LBFGS":
    optimizer = optim.LBFGS(
        list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr
    )
elif CONFIG.optimizer == "SGD":
    optimizer = optim.SGD(
        list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr
    )

scheduler = lr_scheduler.StepLR(
    optimizer, step_size=CONFIG.lr_decay, gamma=CONFIG.gamma
)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(CONFIG.data_variable_size)
tril_indices = get_tril_offdiag_indices(CONFIG.data_variable_size)

if CONFIG.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.DoubleTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if CONFIG.cuda:
        log_prior = log_prior.cuda()

if CONFIG.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


prox_plus = torch.nn.Threshold(0.0, 0.0)


def stau(w, tau):
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def update_optimizer(optimizer, original_lr, c_A):
    """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group["lr"] = lr

    return optimizer, lr


# ===================================
# training:
# ===================================


def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()

    # update optimizer
    optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

    for batch_idx, (data, relations) in enumerate(train_loader):

        if CONFIG.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()

        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        (
            enc_x,
            logits,
            origin_A,
            adj_A_tilt_encoder,
            z_gap,
            z_positive,
            myA,
            Wa,
        ) = encoder(
            data, rel_rec, rel_send
        )  # logits is of size: [num_sims, z_dims]
        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(
            data,
            edges,
            CONFIG.data_variable_size * CONFIG.x_dims,
            rel_rec,
            rel_send,
            origin_A,
            adj_A_tilt_encoder,
            Wa,
        )

        if torch.sum(output != output):
            print("nan error\n")

        target = data
        preds = output
        variance = 0.0

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term
        if CONFIG.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if CONFIG.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += 0.1 * (
                lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap
            )

        # compute h(A)
        h_A = _h_A(origin_A, CONFIG.data_variable_size)
        loss += (
            lambda_A * h_A
            + 0.5 * c_A * h_A * h_A
            + 100.0 * torch.trace(origin_A * origin_A)
            + sparse_loss
        )  # +  0.01 * torch.sum(variance * variance)

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, CONFIG.tau_A * lr)

        if torch.sum(origin_A != origin_A):
            print("nan error\n")

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < CONFIG.graph_threshold] = 0

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print(
        "Epoch: {:04d}".format(epoch),
        "nll_train: {:.10f}".format(np.mean(nll_train)),
        "kl_train: {:.10f}".format(np.mean(kl_train)),
        "ELBO_loss: {:.10f}".format(np.mean(kl_train) + np.mean(nll_train)),
        "mse_train: {:.10f}".format(np.mean(mse_train)),
        "time: {:.4f}s".format(time.time() - t),
    )

    if "graph" not in vars():
        print("error on assign")

    return (
        np.mean(np.mean(kl_train) + np.mean(nll_train)),
        np.mean(nll_train),
        np.mean(mse_train),
        graph,
        origin_A,
    )


# ===================================
# main
# ===================================

t_total = time.time()
best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []
# optimizer step on hyparameters
c_A = CONFIG.c_A
lambda_A = CONFIG.lambda_A
h_A_new = torch.tensor(1.0)
h_tol = CONFIG.h_tol
k_max_iter = int(CONFIG.k_max_iter)
h_A_old = np.inf

try:
    for step_k in range(k_max_iter):
        while c_A < 1e20:
            for epoch in range(CONFIG.epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(
                    epoch, best_ELBO_loss, lambda_A, c_A, optimizer
                )
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, CONFIG.data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A *= 10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

except KeyboardInterrupt:
    print("Done!")


# Create binary adjacency matrix using config threshold
matG1 = np.matrix(origin_A.data.clone().numpy())
final_df = pd.DataFrame(matG1, index=CONFIG.column_names, columns=CONFIG.column_names)

for column in CONFIG.column_names:
    final_df[column] = np.where(np.abs(final_df[column]) < CONFIG.graph_threshold, 0, 1)

# Save final binary adjacency matrix
final_df.to_csv("results/final_adjacency_matrix.csv", index=True)

# Draw the DAG
final_DAG = from_numpy_matrix(final_df.to_numpy(), create_using=nx.DiGraph)
final_DAG = nx.relabel_nodes(
    final_DAG, dict(zip(list(range(CONFIG.data_variable_size)), CONFIG.column_names))
)

final_DAG.remove_nodes_from(list(nx.isolates(final_DAG)))

nx.draw(
    final_DAG,
    node_color="lightcoral",
    node_size=75,
    font_size=3,
    width=0.5,
    arrowsize=4,
    with_labels=True,
    pos=nx.spring_layout(final_DAG),
)
plt.draw()
plt.savefig(os.path.expanduser("results/DAG_plot.png"), format="PNG", dpi=500)
