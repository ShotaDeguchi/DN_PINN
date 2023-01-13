"""
********************************************************************************
utility
********************************************************************************
"""

import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def make_logger(add_data=None):
    now = datetime.datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")

    f_path = "./results/"
    f_name = now + ".txt"
    path = os.path.join(f_path, f_name)

    with open(path, mode="a") as f:
        print(add_data, file=f)
    return path

def write_logger(path, log):
    with open(path, mode="a") as f:
        print(log, file=f)

def plot_comparison(
    epoch, 
    x, y, u_ref, u_inf, u_err, 
    umin, umax,
    vmin, vmax,
    xmin, xmax, xlabel, 
    ymin, ymax, ylabel
):
    uticks = (umax - umin) / 4.
    vticks = (vmax - vmin) / 4.
    xticks = (xmax - xmin) / 4.
    yticks = (ymax - ymin) / 4.

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(x, y, c=u_ref, cmap="turbo", vmin=umin, vmax=umax)
    plt.colorbar(ticks=np.arange(umin, umax+1e-6, uticks))
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("reference")

    plt.subplot(1, 3, 2)
    plt.scatter(x, y, c=u_inf, cmap="turbo", vmin=umin, vmax=umax)
    plt.colorbar(ticks=np.arange(umin, umax+1e-6, uticks))
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("inference")

    plt.subplot(1, 3, 3)
    plt.scatter(x, y, c=np.abs(u_err), cmap="turbo", vmin=vmin, vmax=vmax)
    plt.colorbar(ticks=np.arange(0., vmax+1e-6, vticks))
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("inf. - ref.")
    plt.tight_layout()
    plt.savefig("./results/comparison_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()

def plot_loss_curve(
    epoch, 
    epoch_log, 
    loss_glb_log, 
    loss_pde_log, 
    loss_bc_log
):
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_log, loss_glb_log, ls="-",  alpha=.7, label="loss_glb", c="k")
    plt.plot(epoch_log, loss_pde_log, ls="--", alpha=.3, label="loss_pde", c="tab:blue")
    plt.plot(epoch_log, loss_bc_log,  ls="--", alpha=.3, label="loss_bc",  c="tab:green")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xscale("linear")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    plt.savefig("./results/loss_curve_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()

def plot_error_curve(
    epoch, 
    epoch_log, 
    u_l2_log, u_mse_log, u_sem_log, 
    g_l2_log, g_mse_log, g_sem_log
):
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_log, u_l2_log,  ls="-",  lw=1., alpha=.7, c="tab:blue", label="u_l2")
    plt.plot(epoch_log, u_mse_log, ls="--", lw=1., alpha=.7, c="tab:cyan", label="u_mse")
    plt.fill_between(
        epoch_log, 
        np.array(u_mse_log) + np.array(u_sem_log), 
        np.array(u_mse_log) - np.array(u_sem_log), 
        alpha=.3, color="tab:cyan", label="u_sem"
    )
    plt.plot(epoch_log, g_l2_log,  ls="-",  lw=1., alpha=.7, c="tab:red", label="g_l2")
    plt.plot(epoch_log, g_mse_log, ls="--", lw=1., alpha=.7, c="tab:pink", label="g_mse")
    plt.fill_between(
        epoch_log, 
        np.array(g_mse_log) + np.array(g_sem_log), 
        np.array(g_mse_log) - np.array(g_sem_log), 
        alpha=.3, color="tab:pink", label="g_sem"
    )
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.xscale("linear")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    plt.savefig("./results/error_curve_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()

def plot_grad_dist(
    epoch,
    model: tf.keras.Model, 
    depth: int,
    x_pde, y_pde, g_pde, 
    x_nth, y_nth, u_nth, 
    x_sth, y_sth, u_sth, 
    x_est, y_est, u_est, 
    x_wst, y_wst, u_wst
):
    grad_dist_pde = tf.zeros(shape=(0))
    grad_dist_bc  = tf.zeros(shape=(0))
    for l in range(depth):
        with tf.GradientTape(persistent=True) as tp:
            loss_pde = model.loss_pde(x_pde, y_pde, g_pde)
            loss_nth = model.loss_bc (x_nth, y_nth, u_nth)
            loss_sth = model.loss_bc (x_sth, y_sth, u_sth)
            loss_est = model.loss_bc (x_est, y_est, u_est)
            loss_wst = model.loss_bc (x_wst, y_wst, u_wst)
            loss_bc  = (loss_nth + loss_sth + loss_est + loss_wst) / 4.
        # grad to weights
        grad_pde = tp.gradient(loss_pde, model._weights[l])
        grad_bc  = tp.gradient(loss_bc,  model._weights[l])
        grad_dist_pde = tf.concat([grad_dist_pde, tf.reshape(grad_pde, [-1])], axis=0)
        grad_dist_bc  = tf.concat([grad_dist_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
        del tp
    # cummulated histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(grad_dist_pde, stat="density", element="step", fill=False, alpha=.7, label="grad_pde", color="tab:blue")
    sns.histplot(grad_dist_bc,  stat="density", element="step", fill=False, alpha=.7, label="grad_bc",  color="tab:green")
    plt.xlabel("gradient")
    plt.ylabel("density")
    plt.xscale("linear")
    plt.yscale("symlog")
    plt.xticks(ticks=np.arange(-3., 3., .5))
    plt.xlim(-1., 1.)
    plt.ylim(0., 10 ** 2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("./results/grad_dist_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()

