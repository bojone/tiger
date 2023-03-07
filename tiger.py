# -*- coding: utf-8 -*-
#
# Tiger Optimizer
# (Test passed on tensorflow 1.15)
#
# Tiger only uses momentum and weight decay to update parameters,
# which means that we can achieve gradient accuclation without
# adding a new set of cache parameters. Therefore, Tiger has
# achieved the optimal memory if we have gradient accuclation
# requirements.
#
# Tiger is adapted from Lion and SignSGD. Compared with Lion,
# we simplified the hyperparameters (beta1=beta2) and added an
# adaptive learning rate similar to LAMB. At the same time, we
# propose a simple strategy to prevent the model from diverging
# to NaN (especially when training with mixed precision).
#
# Reference 1: https://kexue.fm/archives/9512
# Reference 2: https://github.com/bojone/tiger
# Reference 3: https://kexue.fm/archives/8634
#

import tensorflow as tf
import re


def root_mean_square(x, axis=None, keepdims=False):
    """Root Mean Square
    """
    return tf.sqrt(tf.reduce_mean(x**2, axis=axis, keepdims=keepdims))


def piecewise_linear(t, schedule, from_zero=True):
    """Piecewise Linear Function
    schedule: a dict like {1000: 1, 2000: 0.1}, which
              means the output will increase linearly
              from 0 to 1 while t ∈ [0, 1000], then
              decrease linearly for 1 to 0.1 while
              t ∈ [1000, 2000], and keep 0.1 while
              t > 2000.
    from_zero: True or False, which mean whether
               promise the schedule start from zero
               or not.
    """
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = tf.cast(t, tf.float32)
    x = tf.cast(schedule[0][1], tf.float32)
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = tf.cast(schedule[i][1], tf.float32)
        x = tf.where(t >= t_begin, x, x_begin)

    return x


class Tiger(tf.keras.optimizers.Optimizer):
    """Tiger Optimizer
    Link1: https://kexue.fm/archives/9512
    Link2: https://github.com/bojone/tiger
    Link3: https://kexue.fm/archives/8634
    """
    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.965,
        weight_decay=0.01,
        grad_accum_steps=1,
        lr_schedule={0: 1},
        shrink_ratio=0.99,
        name='tiger',
        **kwargs
    ):
        super(Tiger, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.grad_accum_steps = grad_accum_steps
        self.lr_schedule = {int(i): j for i, j in lr_schedule.items()}
        self.shrink_ratio = shrink_ratio

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')

    def _prepare(self, var_list):
        t = self.iterations
        b = self.beta
        d = self.weight_decay
        k = self.grad_accum_steps
        s = self.shrink_ratio
        b1 = tf.where(tf.equal(t % k, 0), b, tf.ones_like(b))
        b2 = (1 - b) / k
        lr = self.learning_rate * piecewise_linear(t, self.lr_schedule)
        lr = tf.where(tf.equal((t + 1) % k, 0), lr, tf.zeros_like(lr))
        self._coefficients = (t, d, k, s, b1, b2, lr)
        return super(Tiger, self)._prepare(var_list)

    def _resource_apply(self, grad, var, indices=None):
        t, d, k, s, b1, b2, lr = [
            tf.cast(x, var.dtype) for x in self._coefficients
        ]
        is_nan = tf.reduce_any(tf.is_nan(grad))
        b1 = tf.where(is_nan, tf.ones_like(b1), b1)
        g = tf.where(is_nan, tf.zeros_like(grad), grad)
        m = self.get_slot(var, 'm')

        c = 0
        if re.findall('bias|beta|gamma', var.name):
            lr, d = lr * 0.5, 0
            if 'gamma' in var.name:
                c = 1
        elif 'embeddings' in var.name:
            lr = lr * root_mean_square(var, axis=-1, keepdims=True)
        else:
            lr = lr * root_mean_square(var)

        if indices is None:
            m_t = tf.assign(m, b1 * m + b2 * g)
        else:
            with tf.control_dependencies([tf.assign(m, b1 * m)]):
                m_t = self._resource_scatter_add(m, indices, b2 * g)

        with tf.control_dependencies([m_t]):
            u = (tf.sign(m_t) + d * var) * lr
            v = tf.where(is_nan, (var - c) * s + c, var - u)
            var_t = tf.assign(var, v)
        return var_t

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self.learning_rate,
            'beta': self.beta,
            'weight_decay': self.weight_decay,
            'grad_accum_steps': self.grad_accum_steps,
            'lr_schedule': self.lr_schedule,
            'shrink_ratio': self.shrink_ratio,
        }
        base_config = super(Tiger, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
