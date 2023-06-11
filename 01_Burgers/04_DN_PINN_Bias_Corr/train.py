"""
********************************************************************************
training
********************************************************************************
"""

import os
import time
import yaml
import argparse
from scipy import io
import numpy as np
import tensorflow as tf

from config_gpu import *
from pinn import *
from utils import *

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", 
        "--epochs", 
        type=int, 
        default=300, 
        help="number of epochs"
    )
    parser.add_argument(
        "-b", 
        "--batch_size", 
        type=int, 
        default=-1, 
        help="batch size (-1 for full-batch)"
    )
    parser.add_argument(
        "-p", 
        "--patience", 
        type=int, 
        default=100, 
        help="early stopping patience"
    )
    args = parser.parse_args()
    return args

def main(args):
    # read settings
    with open("./settings.yaml", mode="r") as f:
        settings = yaml.safe_load(f)

    # seed
    seed = settings["SEED"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # prepare logger
    add_data = "seed: " + str(seed)
    logger_path = make_logger(add_data)

    # define a domain
    data = io.loadmat("../reference_Burgers.mat")
    t = data["t"].flatten()[:,None]
    x = data["x"].flatten()[:,None]
    tmin, tmax, nt = t.min(), t.max(), t.shape[0]
    xmin, xmax, nx = x.min(), x.max(), x.shape[0]

    # bounds
    in_lb = tf.constant([tmin, xmin], dtype=tf.float32)
    in_ub = tf.constant([tmax, xmax], dtype=tf.float32)
    in_mean = tf.reduce_mean([in_lb, in_ub], axis=0)

    # sample training points
    N_ic  = int(1e2)
    N_bc  = int(2e2)
    N_pde = int(1e4)

    # PDE
    t_pde = tf.random.uniform((N_pde, 1), in_lb[0], in_ub[0], dtype=tf.float32)
    x_pde = tf.random.uniform((N_pde, 1), in_lb[1], in_ub[1], dtype=tf.float32)
    g_pde = tf.zeros_like(t_pde)   # this is not solution u, but govering eq residual

    # IC
    t_ic = in_lb[0] * tf.ones((N_ic, 1), dtype=tf.float32)
    x_ic = tf.random.uniform((N_ic, 1), in_lb[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_ic = - tf.sin(np.pi * x_ic)

    # BC1 (u(t, x=-1) = 0)
    t_bc1 = tf.random.uniform((int(N_bc / 2), 1), in_lb[0], in_ub[0], dtype=tf.float32, seed=seed)
    x_bc1 = tf.random.uniform((int(N_bc / 2), 1), in_lb[1], in_lb[1], dtype=tf.float32, seed=seed)
    u_bc1 = tf.zeros_like(t_bc1, dtype=tf.float32)

    # BC2 (u(t, x= 1) = 0)
    t_bc2 = tf.random.uniform((int(N_bc / 2), 1), in_lb[0], in_ub[0], dtype=tf.float32, seed=seed)
    x_bc2 = tf.random.uniform((int(N_bc / 2), 1), in_ub[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_bc2 = tf.zeros_like(t_bc2, dtype=tf.float32)

    # reference
    t_ref, x_ref = np.meshgrid(t, x)
    t_ref = t_ref.reshape(-1, 1)
    x_ref = x_ref.reshape(-1, 1)
    t_ref = tf.cast(t_ref, dtype=tf.float32)
    x_ref = tf.cast(x_ref, dtype=tf.float32)
    u_ref = data["usol"]
    u_ref = np.real(u_ref)
    u_ref = u_ref.reshape(-1, 1)

    # define a model
    f_in  = settings["NET_ARCH"]["f_in"]
    f_out = settings["NET_ARCH"]["f_out"]
    f_hid = settings["NET_ARCH"]["f_hid"]
    depth = settings["NET_ARCH"]["depth"]
    lr    = settings["PARAM_TRAIN"]["lr"]
    beta  = settings["PARAM_TRAIN"]["beta"]
    tau   = settings["PARAM_TRAIN"]["tau"]
    model = PINN(
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, 
        w_init="Glorot", b_init="zeros", act="tanh", lr=lr, beta=beta, seed=seed
    )

    # log
    epoch_log = []      # counter
    loss_glb_log = []   # loss function
    loss_pde_log = []
    loss_ic_log  = []
    loss_bc_log  = []
    u_l2_log  = []      # rel. l2 error
    u_mse_log = []      # mse
    u_sem_log = []      # sem
    g_l2_log  = []
    g_mse_log = []
    g_sem_log = []
    gamma_ic_log = []   # biased estimator
    gamma_bc_log = []
    gamma_ic_hat_log = []   # unbiased estimator
    gamma_bc_hat_log = []

    # training
    wait = 0
    loss_best = 9999.
    loss_save = 9999.
    t0 = time.perf_counter()
    for epoch in range(0, args.epochs+1):
        # update gamma
        if epoch % tau == 0:
            _grad_pde, _grad_ic, _grad_bc = model.update_gamma(
                tf.cast(epoch, dtype=tf.float32), tf.cast(tau, dtype=tf.float32),
                t_pde, x_pde, g_pde, 
                t_ic,  x_ic,  u_ic, 
                t_bc1, x_bc1, u_bc1, 
                t_bc2, x_bc2, u_bc2
            )

        # gradient descent
        loss_glb, loss_pde, loss_ic, loss_bc = model.train(
            t_pde, x_pde, g_pde, 
            t_ic,  x_ic,  u_ic, 
            t_bc1, x_bc1, u_bc1, 
            t_bc2, x_bc2, u_bc2
        )

        # log loss
        loss_glb = loss_pde + loss_ic + loss_bc

        epoch_log.append(epoch)
        loss_glb_log.append(loss_glb)
        loss_pde_log.append(loss_pde)
        loss_ic_log.append(loss_ic)
        loss_bc_log.append(loss_bc)

        # log gamma
        gamma_ic = model.gamma_ic.numpy()
        gamma_bc = model.gamma_bc.numpy()
        gamma_ic_log.append(gamma_ic)
        gamma_bc_log.append(gamma_bc)

        # log gamma_hat
        gamma_ic_hat = model.gamma_ic_hat.numpy()
        gamma_bc_hat = model.gamma_bc_hat.numpy()
        gamma_ic_hat_log.append(gamma_ic_hat)
        gamma_bc_hat_log.append(gamma_bc_hat)

        # inference
        u_, g_ = model.infer(t_ref, x_ref)
        u_, g_ = u_.numpy(), g_.numpy()
        u_err = u_ - u_ref
        u_l2  = np.linalg.norm(u_err, ord=2) / np.linalg.norm(u_ref, ord=2)
        u_mse = np.mean(np.square(u_err))
        u_sem = np.std (np.square(u_err), ddof=1) / np.sqrt(float(u_err.shape[0]))
        u_l2_log .append(u_l2)
        u_mse_log.append(u_mse)
        u_sem_log.append(u_sem)
        g_err = g_
        g_l2  = np.linalg.norm(g_err, ord=2)
        g_mse = np.mean(np.square(g_err))
        g_sem = np.std (np.square(g_err), ddof=1) / np.sqrt(float(g_err.shape[0]))
        g_l2_log .append(g_l2)
        g_mse_log.append(g_mse)
        g_sem_log.append(g_sem)

        # print
        t1 = time.perf_counter()
        elps = t1 - t0
        logger_data = \
            f"epoch: {epoch:d}, " \
            f"loss_glb: {loss_glb:.3e}, " \
            f"loss_pde: {loss_pde:.3e}, " \
            f"loss_ic: {loss_ic:.3e}, " \
            f"loss_bc: {loss_bc:.3e}, " \
            f"u_l2: {u_l2:.3e}, " \
            f"g_l2: {g_l2:.3e}, " \
            f"gamma_ic: {gamma_ic:.3e}, " \
            f"gamma_ic_hat: {gamma_ic_hat:.3e}, " \
            f"gamma_bc: {gamma_bc:.3e}, " \
            f"gamma_bc_hat: {gamma_bc_hat:.3e}, " \
            f"loss_best: {loss_best:.3e}, " \
            f"wait: {wait:d}, " \
            f"elps: {elps:.3f}"
        print(logger_data)
        write_logger(logger_path, logger_data)

        # for a fair comparison with SA-PINN (McClenny+2021)
        loss_glb = u_l2
        # save
        if epoch % 250 == 0:
            print(">>>>> saving")
            model.save_weights("./saved_weights/weights_ep" + str(epoch))
            if loss_glb < loss_save:
                print(">>>>> saving")
                model.save_weights("./best_weights/best_weights")
                loss_save = loss_glb

        # early stopping
        if loss_glb < loss_best:
            loss_best = loss_glb
            wait = 0
        else:
            if wait >= args.patience:
                print(">>>>> early stopping")
                break
            wait += 1

        # monitor
        if epoch % 1000 == 0:
            plot_comparison(
                epoch, 
                x=t_ref, y=x_ref, u_ref=u_ref, u_inf=u_, u_err=u_err, 
                umin=-1., umax=1.,
                vmin= 0., vmax=.05,
                xmin=tmin, xmax=tmax, xlabel=r"$t$", 
                ymin=xmin, ymax=xmax, ylabel=r"$x$"
            )
            plot_loss_curve(
                epoch, 
                epoch_log, 
                loss_glb_log, 
                loss_pde_log, 
                loss_ic_log, 
                loss_bc_log
            )
            plot_error_curve(
                epoch, 
                epoch_log, 
                u_l2_log, u_mse_log, u_sem_log, 
                g_l2_log, g_mse_log, g_sem_log
            )
            plot_gamma_curve(
                epoch, 
                epoch_log, 
                gamma_ic_log, gamma_ic_hat_log,
                gamma_bc_log, gamma_bc_hat_log
            )
            plot_grad_dist(
                epoch,
                model, 
                depth,
                t_pde, x_pde, g_pde, 
                t_ic,  x_ic,  u_ic, 
                t_bc1, x_bc1, u_bc1, 
                t_bc2, x_bc2, u_bc2
            )



if __name__ == "__main__":
    args = parse_option()
    config_gpu(flag=0)
    main(args)
