"""
********************************************************************************
Author: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021)
Physics-Informed Neural Network (Raissi+2019)
********************************************************************************
"""

import os
import numpy as np
import tensorflow as tf

class PINN(tf.keras.Model):
    def __init__(
        self, 
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, 
        w_init="Glorot", b_init="zeros", act="tanh", lr=1e-3, seed=42
    ):
        super().__init__()
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.lb     = in_lb      # lower bound of input
        self.ub     = in_ub      # upper bound of input
        self.mean   = in_mean    # mean of input
        self.w_init = w_init     # weight initialization
        self.b_init = b_init     # bias initialization
        self.act    = act        # activation
        self.lr     = lr
        self.seed   = seed
        self.f_scl  = "minmax"   # "linear" / "minmax" / "mean"
        self.l_laaf = False      # L-LAAF (Jagtap+2020)
        self.g_enhc = False      # Gradient Enhancement (Yu+2022)
        self.d_type = tf.float32

        # seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # build a network
        self._layers = [self.f_in] + (self.depth - 1) * [self.f_hid] + [self.f_out]
        self._weights, self._biases, self._alphas, self._params \
            = self.dnn_initializer(self._layers)

        # optimizer (overwrite the learning rate if necessary)
        # self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3, decay_steps=1000, decay_rate=.9
        # )
        # self.lr = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=1e-3, decay_steps=1000, alpha=0.0
        # )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # system params
        self.nu = tf.constant(.01 / np.pi, dtype=self.d_type)

        # hello
        print("***************************************************************")
        print("************************* DELLO WORLD *************************")
        print("***************************************************************")

    def dnn_initializer(self, layers):
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for l in range(0, self.depth):
            w = self.weight_initializer(shape=[layers[l], layers[l+1]], depth=l)
            b = self.bias_initializer  (shape=[       1,  layers[l+1]], depth=l)
            weights.append(w)
            biases.append(b)
            params.append(w)
            params.append(b)
            if self.l_laaf == True and l < self.depth - 1:
                a = tf.Variable(1., dtype=self.d_type, name="a"+str(l))
                alphas.append(a)
                params.append(a)
            elif self.l_laaf == False and l < self.depth - 1:
                a = tf.constant(1., dtype=self.d_type, name="a"+str(l))
                alphas.append(a)
        return weights, biases, alphas, params

    def weight_initializer(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.w_init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif self.w_init == "He":
            std = np.sqrt(2 / in_dim)
        elif self.w_init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedError(">>>>> weight_initializer")
        weight = tf.Variable(
            tf.random.truncated_normal(shape = [in_dim, out_dim], \
            mean = 0., stddev = std, dtype = self.d_type), \
            dtype = self.d_type, name = "w" + str(depth)
            )
        return weight

    def bias_initializer(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.b_init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape = [in_dim, out_dim], dtype = self.d_type), \
                dtype = self.d_type, name = "b" + str(depth)
                )
        elif self.b_init == "ones":
            bias = tf.Variable(
                tf.ones(shape = [in_dim, out_dim], dtype = self.d_type), \
                dtype = self.d_type, name = "b" + str(depth)
                )
        else:
            raise NotImplementedError(">>>>> bias_initializer")
        return bias

    def forward_pass(self, x):
        # feature scaling
        if self.f_scl == None or self.f_scl == "linear":
            z = tf.identity(x)
        elif self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mean) / (self.ub - self.lb)
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scl)")

        # forward pass
        for l in range(0, self.depth - 1):
            w = self._weights[l]
            b = self._biases [l]
            a = self._alphas [l]
            u = tf.math.add(tf.linalg.matmul(z, w), b)
            u = tf.multiply(a, u)
            if self.act == "tanh":
                z = tf.math.tanh(u)
            elif self.act == "softplus":
                z = tf.math.softplus(u)
            elif self.act == "silu" or self.act == "swish":
                z = tf.multiply(u, tf.math.sigmoid(u))
            elif self.act == "gelu":
                z = tf.multiply(u, tf.math.sigmoid(1.702 * u))
            elif self.act == "mish":
                z = tf.multiply(u, tf.math.tanh(tf.math.softplus(u)))
            else:
                raise NotImplementedError(">>>>> forward_pass (act)")
        w = self._weights[-1]
        b = self._biases [-1]
        u = tf.math.add(tf.linalg.matmul(z, w), b)
        z = tf.identity(u)   # identity
        y = tf.identity(z)
        return y

    def compute_pde(self, t, x):
        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch(t)
            tp1.watch(x)
            with tf.GradientTape(persistent=True) as tp2:
                tp2.watch(t)
                tp2.watch(x)
                u = self.forward_pass(tf.concat([t, x], axis=1))
            u_x = tp2.gradient(u, x)
            del tp2
        u_t  = tp1.gradient(u, t)
        u_xx = tp1.gradient(u_x, x)
        del tp1
        g = u_t + u * u_x - self.nu * u_xx
        return u, g

    @tf.function
    def loss_pde(
        self, 
        t, x, g
    ):
        u_, g_ = self.compute_pde(t, x)
        loss = tf.reduce_mean(tf.square(g_))
        return loss

    @tf.function
    def loss_ge(
        self, 
        t, x, g, weight
    ):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(t)
            tp.watch(x)
            u_, g_ = self.compute_pde(t, x)
        g_t_ = tp.gradient(g_, t)
        g_x_ = tp.gradient(g_, x)
        del tp
        loss = tf.reduce_mean(tf.square(g_t_)) \
                + tf.reduce_mean(tf.square(g_x_))
        loss *= weight
        return loss

    @tf.function
    def loss_ic(
        self, 
        t, x, u
    ):
        u_, g_ = self.compute_pde(t, x)
        loss = tf.reduce_mean(tf.square(u_ - u))
        return loss

    @tf.function
    def loss_bc(
        self, 
        t, x, u
    ):
        u_, g_ = self.compute_pde(t, x)
        loss = tf.reduce_mean(tf.square(u_ - u))
        return loss

    @tf.function
    def train(
        self, 
        t_pde, x_pde, g_pde, 
        t_ic,  x_ic,  u_ic, 
        t_bc1, x_bc1, u_bc1, 
        t_bc2, x_bc2, u_bc2
    ):
        with tf.GradientTape(persistent=True) as tp:
            loss_pde = self.loss_pde(t_pde, x_pde, g_pde)
            loss_ic  = self.loss_ic (t_ic,  x_ic,  u_ic)
            loss_bc1 = self.loss_bc (t_bc1, x_bc1, u_bc1)
            loss_bc2 = self.loss_bc (t_bc2, x_bc2, u_bc2)
            loss_bc  = (loss_bc1 + loss_bc2) / 2.
            loss_glb = loss_pde \
                        + loss_ic \
                        + loss_bc
            if self.l_laaf == True:   # L-LAAF (Jagtap+2020)
                loss_glb += 1. / tf.reduce_mean(tf.exp(self._alphas))
            if self.g_enhc == True:   # Gradient Enhancement (Yu+2022) (weight \in [1e-5, 1e-2])
                loss_glb += self.loss_ge(t_pde, x_pde, g_pde, weight=1e-4)
        grad = tp.gradient(loss_glb, self._params)
        del tp
        self.optimizer.apply_gradients(zip(grad, self._params))
        return loss_glb, loss_pde, loss_ic, loss_bc

    def infer(
        self,
        t, x
    ):
        u_, gv_ = self.compute_pde(t, x)
        return u_, gv_
