"""
********************************************************************************
training
********************************************************************************
"""

import os
import time
import yaml
import argparse
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
    xmin, xmax, nx = 0., 1., 201
    ymin, ymax, ny = 0., 1., 201
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    x, y = np.meshgrid(x, y)

    # reference
    scale = 4.
    omega = scale * np.pi
    def u_ref(x, y, omgea):
        u = np.cos(omgea * x) * np.sin(omgea * y)
        return u
    x_ref, y_ref = x.reshape(-1, 1), y.reshape(-1, 1)
    x_ref, y_ref = tf.cast(x_ref, dtype=tf.float32), tf.cast(y_ref, dtype=tf.float32)
    u_ref = u_ref(x_ref, y_ref, omega).reshape(-1, 1)

    # bounds
    in_lb = tf.constant([xmin, ymin], dtype=tf.float32)
    in_ub = tf.constant([xmax, ymax], dtype=tf.float32)
    in_mean = tf.reduce_mean([in_lb, in_ub], axis=0)

    # sample training points
    N_pde = int(2500)
    N_bc  = int( 100)   # 100 / each NSEW bounds -> 400 in total

    # PDE
    x_pde = tf.random.uniform((N_pde, 1), in_lb[0], in_ub[0], dtype=tf.float32)
    y_pde = tf.random.uniform((N_pde, 1), in_lb[1], in_ub[1], dtype=tf.float32)
    g_pde = tf.zeros_like(x_pde)   # this is not solution u, but govering eq residual
    # north bound
    x_nth = tf.random.uniform((N_bc, 1), in_lb[0], in_ub[0], dtype=tf.float32, seed=seed)
    y_nth = tf.random.uniform((N_bc, 1), in_ub[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_nth = tf.cos(omega * x_nth) * tf.sin(omega * y_nth)
    # south
    x_sth = tf.random.uniform((N_bc, 1), in_lb[0], in_ub[0], dtype=tf.float32, seed=seed)
    y_sth = tf.random.uniform((N_bc, 1), in_lb[1], in_lb[1], dtype=tf.float32, seed=seed)
    u_sth = tf.cos(omega * x_sth) * tf.sin(omega * y_sth)
    # east
    x_est = tf.random.uniform((N_bc, 1), in_ub[0], in_ub[0], dtype=tf.float32, seed=seed)
    y_est = tf.random.uniform((N_bc, 1), in_lb[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_est = tf.cos(omega * x_est) * tf.sin(omega * y_est)
    # west
    x_wst = tf.random.uniform((N_bc, 1), in_lb[0], in_lb[0], dtype=tf.float32, seed=seed)
    y_wst = tf.random.uniform((N_bc, 1), in_lb[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_wst = tf.cos(omega * x_wst) * tf.sin(omega * y_wst)

    # define a model
    f_in  = settings["NET_ARCH"]["f_in"]
    f_out = settings["NET_ARCH"]["f_out"]
    f_hid = settings["NET_ARCH"]["f_hid"]
    depth = settings["NET_ARCH"]["depth"]
    lr    = 1e-3
    model = PINN(
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, omega, 
        w_init="Glorot", b_init="zeros", act="tanh", lr=lr, seed=seed
    )

    # log
    epoch_log = []      # counter
    loss_glb_log = []   # loss function
    loss_pde_log = []
    loss_bc_log  = []
    u_l2_log  = []   # rel. l2 error
    u_mse_log = []   # mse
    u_sem_log = []   # sem
    g_l2_log  = []
    g_mse_log = []
    g_sem_log = []

    # training
    wait = 0
    loss_best = 9999.
    loss_save = 9999.
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        # gradient descent
        loss_glb, loss_pde, loss_bc = model.train(
            x_pde, y_pde, g_pde, 
            x_nth, y_nth, u_nth, 
            x_sth, y_sth, u_sth, 
            x_est, y_est, u_est, 
            x_wst, y_wst, u_wst
        )

        # log
        epoch_log.append(epoch)
        loss_glb_log.append(loss_glb)
        loss_pde_log.append(loss_pde)
        loss_bc_log.append(loss_bc)

        # inference
        u_, g_ = model.infer(x_ref, y_ref)
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
            f"loss_bc: {loss_bc:.3e}, " \
            f"u_l2: {u_l2:.3e}, " \
            f"g_l2: {g_l2:.3e}, " \
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
                x=x_ref, y=y_ref, u_ref=u_ref, u_inf=u_, u_err=u_err, 
                umin=-1., umax=1.,
                vmin= 0., vmax=.05,
                xmin=xmin, xmax=xmax, xlabel="x", 
                ymin=ymin, ymax=ymax, ylabel="y"
            )
            plot_loss_curve(
                epoch, 
                epoch_log, 
                loss_glb_log, 
                loss_pde_log, 
                loss_bc_log
            )
            plot_error_curve(
                epoch, 
                epoch_log, 
                u_l2_log, u_mse_log, u_sem_log, 
                g_l2_log, g_mse_log, g_sem_log
            )
            plot_grad_dist(
                epoch,
                model, 
                depth,
                x_pde, y_pde, g_pde, 
                x_nth, y_nth, u_nth, 
                x_sth, y_sth, u_sth, 
                x_est, y_est, u_est, 
                x_wst, y_wst, u_wst
            )



if __name__ == "__main__":
    args = parse_option()
    config_gpu(flag=0)
    main(args)
