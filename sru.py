#coding: utf-8
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class SRUCell(RNNCell):
    def __init__(self, num_units):        
        super(SRUCell, self).__init__()
        self._num_units = num_units

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):                        
            c, _ = state
            x_size = x.get_shape().as_list()[1]
            
            W_u = tf.get_variable('W_u', [x_size, 3 * self.output_size])
            b_f = tf.get_variable('b_f', [self._num_units])
            b_r = tf.get_variable('b_r', [self._num_units])

            xh = tf.matmul(x, W_u)
            z, f, r = tf.split(1, 3, xh)            

            f = tf.sigmoid(f + b_f)
            r = tf.sigmoid(r + b_r)            

            new_c = f * c + (1 - f) * z
            new_h = r * tf.tanh(new_c)

            using_highway = (x_size == self._num_units)
            if using_highway:
                new_h += (1 - r) * x

            return new_h, (new_c, new_h)
