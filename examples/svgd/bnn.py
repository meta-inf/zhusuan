#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
from zhusuan.variational import svgd

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            ws.append(
                zs.Normal('w' + str(i), w_mu, std=1.,
                          n_samples=n_particles, group_ndims=2))

        # forward
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, logstd=y_logstd)

    return model, y_mean


def stein_variational(layer_sizes, n_particles):
    ws = {}
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                          layer_sizes[1:])):
        w_vals = tf.get_variable(
            'w_val' + str(i), shape=[n_particles, 1, n_out, n_in + 1],
            #initializer=tf.random_uniform_initializer(-0.5, +0.5))
            initializer=tf.random_normal_initializer(stddev=1./np.sqrt(n_in)))
        ws['w' + str(i)] = w_vals
    return ws


tf.set_random_seed(1237)
np.random.seed(1234)

# Load UCI Boston housing data
# data_path = os.path.join(conf.data_dir, 'protein_data.data')
# x_train, y_train, x_valid, y_valid, x_test, y_test = \
#     dataset.load_uci_protein_data(data_path)
data_path = os.path.join(conf.data_dir, 'boston.data')
x_train, y_train, x_valid, y_valid, x_test, y_test = \
    dataset.load_uci_boston_housing(data_path)

x_train = np.vstack([x_train, x_valid]).astype('f')
y_train = np.hstack([y_train, y_valid]).astype('f')
N, n_x = x_train.shape

# Standardize data
x_train, x_test, _, _ = dataset.standardize(x_train, x_test.astype('f'))
y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
    y_train, y_test.astype('f'))

# Define model parameters
if data_path.find('protein') != -1:
    n_hiddens = [100]
else:
    n_hiddens = [50]    

# Build the computation graph
n_particles = 20
x = tf.placeholder(tf.float32, shape=[None, n_x])
y = tf.placeholder(tf.float32, shape=[None])
layer_sizes = [n_x] + n_hiddens + [1]
w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

def log_joint(observed):
    observed = observed.copy()
    observed.update({'y': y})
    model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
    log_pws = model.local_log_prob(w_names)
    log_py_xw = model.local_log_prob('y')
    return tf.add_n(log_pws) + log_py_xw * N

variational = stein_variational(layer_sizes, n_particles)
grad_and_vars = zs.variational.svgd.stein_variational_gradient(
    log_joint, variational)
grad_and_vars = [(-g, v) for (g, v) in grad_and_vars]
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
infer_op = optimizer.apply_gradients(grad_and_vars)

# prediction: rmse & log likelihood
observed = variational.copy()
observed.update({'y': y})
model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
y_pred = tf.reduce_mean(y_mean, 0)
rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
log_py_xw = model.local_log_prob('y')
log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
    tf.log(std_y_train)

# Define training/evaluation parameters
lb_samples = 10
ll_samples = 5000
epochs = 5000
batch_size = 100
iters = int(np.floor(x_train.shape[0] / float(batch_size)))
test_freq = 10

par = variational['w0']
par = tf.reshape(par, [tf.shape(par)[0], -1])
len_par = tf.cast(tf.shape(par)[1], tf.float32)
par_sqdist = tf.reduce_sum(
    (tf.expand_dims(par, 0) - tf.expand_dims(par, 1))**2,
    axis=-1)
avg_dist = (tf.reduce_sum(par_sqdist)) / \
    tf.cast(n_particles * (n_particles-1), tf.float32)

# Run the inference
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(1, epochs + 1):
    lbs = []
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    avg_dist_ = sess.run(
        avg_dist, feed_dict={x: x_test, y: y_test})
    for t in range(iters):
        x_batch = x_train[indices[t * batch_size:(t + 1) * batch_size]]
        y_batch = y_train[indices[t * batch_size:(t + 1) * batch_size]]
        _ = sess.run(
            [infer_op],
            feed_dict={x: x_batch, y: y_batch})
    # print('Epoch {} avg dist {}'.format(epoch, avg_dist_))

    if epoch % test_freq == 0:
        test_rmse, test_ll = sess.run(
            [rmse, log_likelihood],
            feed_dict={x: x_test, y: y_test})
        print('>> TEST')
        print('>> Test rmse = {}, log_likelihood = {} avg_dist = {}'
              .format(test_rmse, test_ll, avg_dist_))
