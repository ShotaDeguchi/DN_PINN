"""
********************************************************************************
Author: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021)
Physics-Informed Neural Networks (Raissi+2019)
Dynamic Norm-alization of PINN (Deguchi+)
********************************************************************************
"""

import os
import numpy as np
import tensorflow as tf

class PINN(tf.keras.Model):
    def __init__(
        self, 
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, omega, 
        w_init="Glorot", b_init="zeros", act="tanh", lr=1e-3, beta=.99, seed=42
    ):
        super().__init__()
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.lb     = in_lb
        self.ub     = in_ub
        self.mean   = in_mean
        self.w_init = w_init
        self.b_init = b_init
        self.act    = act
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

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # system params
        self.omega = tf.constant(omega, dtype=self.d_type)

        # Dynamic Normalization of PINN (Deguchi+)
        self.beta = tf.constant(beta, dtype=self.d_type)     # exponential decay rate
        self.gamma_bc = tf.Variable(1., dtype=self.d_type)   # weight associated with loss_bc

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

    def compute_pde(self, x, y):
        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch(x)
            tp1.watch(y)
            with tf.GradientTape(persistent=True) as tp2:
                tp2.watch(x)
                tp2.watch(y)
                u = self.forward_pass(tf.concat([x, y], axis=1))
            u_x = tp2.gradient(u, x)
            u_y = tp2.gradient(u, y)
            del tp2
        u_xx = tp1.gradient(u_x, x)
        u_yy = tp1.gradient(u_y, y)
        del tp1
        src = -2. * self.omega**2 * tf.cos(self.omega * x) * tf.sin(self.omega * y)
        g = u_xx + u_yy - src
        return u, g

    @tf.function
    def loss_pde(
        self, 
        x, y, g
    ):
        u_, g_ = self.compute_pde(x, y)
        loss = tf.reduce_mean(tf.square(g_))
        return loss

    @tf.function
    def loss_ge(
        self, 
        x, y, g, weight
    ):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(x)
            tp.watch(y)
            u_, g_ = self.compute_pde(x, y)
        g_x_ = tp.gradient(g_, x)
        g_y_ = tp.gradient(g_, y)
        del tp
        loss = tf.reduce_mean(tf.square(g_x_)) \
                + tf.reduce_mean(tf.square(g_y_))
        loss *= weight
        return loss

    @tf.function
    def loss_bc(
        self, 
        x, y, u
    ):
        u_, g_ = self.compute_pde(x, y)
        loss = tf.reduce_mean(tf.square(u_ - u))
        return loss

    @tf.function
    def train(
        self, 
        x_pde, y_pde, g_pde, 
        x_nth, y_nth, u_nth, 
        x_sth, y_sth, u_sth, 
        x_est, y_est, u_est, 
        x_wst, y_wst, u_wst
    ):
        with tf.GradientTape(persistent=True) as tp:
            loss_pde = self.loss_pde(x_pde, y_pde, g_pde)
            loss_nth = self.loss_bc (x_nth, y_nth, u_nth)
            loss_sth = self.loss_bc (x_sth, y_sth, u_sth)
            loss_est = self.loss_bc (x_est, y_est, u_est)
            loss_wst = self.loss_bc (x_wst, y_wst, u_wst)
            loss_bc  = (loss_nth + loss_sth + loss_est + loss_wst) / 4.
            loss_glb = loss_pde \
                        + self.gamma_bc * loss_bc
            if self.l_laaf == True:   # L-LAAF (Jagtap+2020)
                loss_glb += 1. / tf.reduce_mean(tf.exp(self._alphas))
            if self.g_enhc == True:   # Gradient Enhancement (Yu+2022) (weight \in [1e-5, 1e-2])
                loss_glb += self.loss_ge(x_pde, y_pde, g_pde, weight=1e-4)
        grad = tp.gradient(loss_glb, self._params)
        del tp
        self.optimizer.apply_gradients(zip(grad, self._params))
        return loss_glb, loss_pde, loss_bc

    def infer(
        self,
        x, y
    ):
        u_, g_ = self.compute_pde(x, y)
        return u_, g_

    @tf.function
    def update_gamma(
        self, 
        x_pde, y_pde, g_pde, 
        x_nth, y_nth, u_nth, 
        x_sth, y_sth, u_sth, 
        x_est, y_est, u_est, 
        x_wst, y_wst, u_wst
    ):
        """
        Updates gamma's (weights associated with loss components) using gradient norms
        """

        _grad_pde = tf.zeros(shape=(0))
        _grad_bc  = tf.zeros(shape=(0))
        for l in range(self.depth):
            with tf.GradientTape(persistent=True) as tp:
                loss_pde = self.loss_pde(x_pde, y_pde, g_pde)
                loss_nth = self.loss_bc (x_nth, y_nth, u_nth)
                loss_sth = self.loss_bc (x_sth, y_sth, u_sth)
                loss_est = self.loss_bc (x_est, y_est, u_est)
                loss_wst = self.loss_bc (x_wst, y_wst, u_wst)
                loss_bc  = (loss_nth + loss_sth + loss_est + loss_wst) / 4.
            # grad to weights
            grad_pde = tp.gradient(loss_pde, self._weights[l])
            grad_bc  = tp.gradient(loss_bc,  self._weights[l])
            _grad_pde = tf.concat([_grad_pde, tf.reshape(grad_pde, [-1])], axis=0)
            _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
            # # grad to biases (skipped, since the last bias is not tracked in loss_pde bc of pde structure)
            # grad_pde = tp.gradient(loss_pde, self._biases[l])
            # grad_bc  = tp.gradient(loss_bc,  self._biases[l])
            # _grad_pde = tf.concat([_grad_pde, tf.reshape(grad_pde, [-1])], axis=0)
            # _grad_bc  = tf.concat([_grad_bc,  tf.reshape(grad_bc,  [-1])], axis=0)
            del tp

        # Lp norm (p = 1, 2, ..., np.inf)
        p = 2
        _norm_pde = tf.norm(_grad_pde, ord=p)
        _norm_bc  = tf.norm(_grad_bc,  ord=p)

        # elongation
        gamma_bc = _norm_pde / _norm_bc

        # exponential decay
        gamma_bc = self.beta * self.gamma_bc  + (1. - self.beta) * gamma_bc

        # assign() to update
        self.gamma_bc.assign(gamma_bc)

        return _norm_pde, _norm_bc
