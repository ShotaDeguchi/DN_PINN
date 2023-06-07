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



if __name__ == "__main__":
    config_gpu(flag=-1)
    main()
