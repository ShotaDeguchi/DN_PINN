"""
********************************************************************************
inference
********************************************************************************
"""

import yaml
from scipy import io
import numpy as np
import tensorflow as tf

from config_gpu import *
from pinn import *
from utils import *

def main():
    # read settings
    with open("./settings.yaml", mode="r") as f:
        settings = yaml.safe_load(f)

    # seed
    seed = settings["SEED"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # define a domain
    data = io.loadmat("../reference_AllenCahn.mat")
    t = data["tt"].flatten()[:,None]
    x = data["x"] .flatten()[:,None]
    tmin, tmax, nt = t.min(), t.max(), t.shape[0]
    xmin, xmax, nx = x.min(), x.max(), x.shape[0]

    # bounds
    in_lb = tf.constant([tmin, xmin], dtype=tf.float32)
    in_ub = tf.constant([tmax, xmax], dtype=tf.float32)
    in_mean = tf.reduce_mean([in_lb, in_ub], axis=0)

    # sample training points
    N_pde = int(2e4)
    N_ic  = int(512)
    N_bc  = int(1e2)

    # PDE
    t_pde = tf.random.uniform((N_pde, 1), in_lb[0], in_ub[0], dtype=tf.float32)
    x_pde = tf.random.uniform((N_pde, 1), in_lb[1], in_ub[1], dtype=tf.float32)
    g_pde = tf.zeros_like(t_pde)   # this is not solution u, but govering eq residual
    # IC
    t_ic = in_lb[0] * tf.ones((N_ic, 1), dtype=tf.float32)
    x_ic = tf.random.uniform((N_ic, 1), in_lb[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_ic = (x_ic ** 2) * tf.cos(np.pi * x_ic)
    # BC1 (u(t, x=-1) = 0)
    t_bc1 = tf.random.uniform((N_bc, 1), in_lb[0], in_ub[0], dtype=tf.float32, seed=seed)
    x_bc1 = tf.random.uniform((N_bc, 1), in_lb[1], in_lb[1], dtype=tf.float32, seed=seed)
    u_bc1 = tf.zeros_like(t_bc1, dtype=tf.float32)
    # BC2 (u(t, x= 1) = 0)
    t_bc2 = tf.identity(t_bc1)
    x_bc2 = tf.random.uniform((N_bc, 1), in_ub[1], in_ub[1], dtype=tf.float32, seed=seed)
    u_bc2 = tf.zeros_like(t_bc2, dtype=tf.float32)

    # reference
    t_ref, x_ref = np.meshgrid(t, x)
    t_ref = t_ref.reshape(-1, 1)
    x_ref = x_ref.reshape(-1, 1)
    t_ref = tf.cast(t_ref, dtype=tf.float32)
    x_ref = tf.cast(x_ref, dtype=tf.float32)
    u_ref = data["uu"]
    u_ref = np.real(u_ref)
    u_ref = u_ref.reshape(-1, 1)

    # define a model
    f_in  = settings["NET_ARCH"]["f_in"]
    f_out = settings["NET_ARCH"]["f_out"]
    f_hid = settings["NET_ARCH"]["f_hid"]
    depth = settings["NET_ARCH"]["depth"]
    model = PINN(
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, seed=seed
    )
    model.load_weights("./best_weights/best_weights")

    # inference
    u_, g_ = model.infer(t_ref, x_ref)
    u_, g_ = u_.numpy(), g_.numpy()
    u_err = u_ - u_ref
    u_l2  = np.linalg.norm(u_err, ord=2) / np.linalg.norm(u_ref, ord=2)
    g_l2  = np.linalg.norm(g_, ord=2)
    u_mse = np.mean(np.square(u_err))
    u_sem = np.std (np.square(u_err), ddof=1) / np.sqrt(float(u_err.shape[0]))
    print("inference result;")
    print("l2: %.6e, mse: %.6e, sem: %.6e" % (u_l2, u_mse, u_sem))

    epoch = "inference"
    plot_comparison(
        epoch, 
        x=t_ref, y=x_ref, u_ref=u_ref, u_inf=u_, u_err=u_err, 
        umin=-1., umax=1.,
        vmin= 0., vmax=.05,
        xmin=tmin, xmax=tmax, xlabel="t", 
        ymin=xmin, ymax=xmax, ylabel="x"
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
    config_gpu(flag=-1)
    main()
