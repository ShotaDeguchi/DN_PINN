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

# visualization setting
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

################################################################################

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

################################################################################

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
    levels = np.linspace(umin, umax, 32)
    ticks = np.linspace(umin, umax, 5)
    plt.contourf(
        tf.reshape(x, [201, 201]),
        tf.reshape(y, [201, 201]),
        tf.reshape(u_ref, [201, 201]),
        cmap="turbo", levels=levels, extend="both"
    )
    plt.colorbar(ticks=ticks)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(r"Reference, $u$")

    plt.subplot(1, 3, 2)
    plt.contourf(
        tf.reshape(x, [201, 201]),
        tf.reshape(y, [201, 201]),
        tf.reshape(u_inf, [201, 201]),
        cmap="turbo", levels=levels, extend="both"
    )
    plt.colorbar(ticks=ticks)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(r"Inference, $\hat{u}$")

    plt.subplot(1, 3, 3)
    levels = np.linspace(vmin, vmax, 32)
    ticks = np.linspace(vmin, vmax, 5)
    plt.contourf(
        tf.reshape(x, [201, 201]),
        tf.reshape(y, [201, 201]),
        tf.reshape(np.abs(u_err), [201, 201]),
        cmap="turbo", levels=levels, extend="both"
    )
    plt.colorbar(ticks=ticks)
    plt.xticks(np.arange(xmin, xmax+1e-6, xticks))
    plt.yticks(np.arange(ymin, ymax+1e-6, yticks))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(r"$| \hat{u} - u |$")
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
    plt.plot(epoch_log, loss_glb_log, ls="-",  alpha=1., c="tab:gray",  label=r"$\mathcal{L}$")
    plt.plot(epoch_log, loss_pde_log, ls="--", alpha=.3, c="tab:blue",  label=r"$\mathcal{L}_{\mathrm{PDE}}$")
    plt.plot(epoch_log, loss_bc_log,  ls="--", alpha=.3, c="tab:green", label=r"$\mathcal{L}_{\mathrm{BC}}$")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    plt.plot(epoch_log, u_l2_log,  ls="-",  lw=1., alpha=.7, c="tab:blue", label=r"$\| \hat{u} - u \|_{2} / \| u \|_{2}$")
    plt.plot(epoch_log, u_mse_log, ls="--", lw=1., alpha=.7, c="tab:cyan", label=r"$\mathrm{MSE}(\hat{u}, u)$")
    plt.fill_between(
        epoch_log, 
        np.array(u_mse_log) + np.array(u_sem_log), 
        np.array(u_mse_log) - np.array(u_sem_log), 
        alpha=.3, color="tab:cyan", label=r"$\mathrm{SE}(\hat{u}, u)$"
    )
    plt.plot(epoch_log, g_l2_log,  ls="-",  lw=1., alpha=.7, c="tab:red", label=r"$\| \hat{g} - 0 \|_{2}$")
    plt.plot(epoch_log, g_mse_log, ls="--", lw=1., alpha=.7, c="tab:pink", label=r"$\mathrm{MSE}(\hat{g}, 0)$")
    plt.fill_between(
        epoch_log, 
        np.array(g_mse_log) + np.array(g_sem_log), 
        np.array(g_mse_log) - np.array(g_sem_log), 
        alpha=.3, color="tab:pink", label=r"$\mathrm{SE}(\hat{g}, 0)$"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    plt.savefig("./results/error_curve_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()

def plot_gamma_curve(
    epoch, 
    epoch_log, 
    gamma_bc_log, gamma_bc_hat_log
):
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_log, gamma_bc_log, ls="-",  lw=1., alpha=.7, c="tab:green", label=r"$\gamma_{\mathrm{BC}}$")
    plt.plot(epoch_log, gamma_bc_hat_log, ls="--", lw=1., alpha=.7, c="tab:green", label=r"$\hat{\gamma}_{\mathrm{BC}}$")
    plt.legend(loc="upper right")
    plt.grid(alpha=.5)
    plt.tight_layout()
    plt.savefig("./results/gamma_curve_" + str(epoch) + ".png", dpi=300)
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
    gamma_bc = model.gamma_bc.numpy()
    gamma_bc_hat = model.gamma_bc_hat.numpy()

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
    sns.histplot(grad_dist_pde, stat="density", element="step", fill=False, alpha=.7, color="tab:blue", label=r"$\nabla_{\theta} \mathcal{L}_{\mathrm{PDE}}$")
    sns.histplot(gamma_bc * grad_dist_bc,  stat="density", element="step", fill=False, alpha=.7, color="tab:green", label=r"$\gamma_{\mathrm{BC}} \nabla_{\theta} \mathcal{L}_{\mathrm{BC}}$")
    sns.histplot(gamma_bc_hat * grad_dist_bc,  stat="density", element="step", fill=False, alpha=.7, ls="--", color="tab:green", label=r"$\hat{\gamma}_{\mathrm{BC}} \nabla_{\theta} \mathcal{L}_{\mathrm{BC}}$")
    plt.xlabel("Gradient")
    plt.ylabel("Density")
    plt.yscale("symlog")
    plt.xticks(ticks=np.arange(-3., 3., .5))
    plt.xlim(-1., 1.)
    plt.ylim(0., 10 ** 2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("./results/grad_dist_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()
