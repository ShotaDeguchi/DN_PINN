"""
********************************************************************************
inference
********************************************************************************
"""

import yaml
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
    xmin, xmax, nx = 0., 1., 201
    ymin, ymax, ny = 0., 1., 201
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    x, y = np.meshgrid(x, y)

    # reference
    scale = 4.
    omega = scale * np.pi
    def u_ref(x, y, omgea):
        u = np.cos(omgea * x)*np.sin(omgea * y)
        return u
    x_ref, y_ref = x.reshape(-1, 1), y.reshape(-1, 1)
    x_ref, y_ref = tf.cast(x_ref, dtype=tf.float32), tf.cast(y_ref, dtype=tf.float32)
    u_ref = u_ref(x_ref, y_ref, omega).reshape(-1, 1)

    # bounds
    in_lb = tf.constant([xmin, ymin], dtype=tf.float32)
    in_ub = tf.constant([xmax, ymax], dtype=tf.float32)
    in_mean = tf.reduce_mean([in_lb, in_ub], axis=0)

    # define a model
    f_in  = settings["NET_ARCH"]["f_in"]
    f_out = settings["NET_ARCH"]["f_out"]
    f_hid = settings["NET_ARCH"]["f_hid"]
    depth = settings["NET_ARCH"]["depth"]
    model = PINN(
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, omega, seed=seed
    )
    model.load_weights("./best_weights/best_weights")

    # inference
    u_, g_ = model.infer(x_ref, y_ref)
    u_, g_ = u_.numpy(), g_.numpy()
    u_err = u_ - u_ref
    u_l2  = np.linalg.norm(u_err, ord=2) / np.linalg.norm(u_ref, ord=2)
    g_l2  = np.linalg.norm(g_,    ord=2)
    u_mse = np.mean(np.square(u_err))
    u_sem = np.std (np.square(u_err), ddof=1) / np.sqrt(float(u_err.shape[0]))
    print("inference result;")
    print("l2: %.6e, mse: %.6e, sem: %.6e" % (u_l2, u_mse, u_sem))

    epoch = "inference"
    plot_comparison(
        epoch, 
        x=x_ref, y=y_ref, u_ref=u_ref, u_inf=u_, u_err=u_err, 
        umin=-1., umax=1.,
        vmin= 0., vmax=.05,
        xmin=xmin, xmax=xmax, xlabel="x", 
        ymin=ymin, ymax=ymax, ylabel="y"
    )



if __name__ == "__main__":
    config_gpu(flag=-1)
    main()
