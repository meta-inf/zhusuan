"""
Random hack. Code baesd on Qiang Liu's original repo.
"""
import tensorflow as tf
import zhusuan as zs


__all__ = ['stein_variational_gradient']


def rbf_kernel(theta_x, theta_y, bandwidth='median'):
    """
    :param theta: tensor of shape [n_particles, n_params]
    :return: tensor of shape [n_particles, n_particles]
    """
    n_x = tf.shape(theta_x)[0]
    pairwise_dists = tf.reduce_sum(
        (tf.expand_dims(theta_x, 1) - tf.expand_dims(theta_y, 0)) ** 2,
        axis=-1)
    if bandwidth == 'median':
        bandwidth = tf.contrib.distributions.percentile(
            tf.squeeze(pairwise_dists), q=50.)
        bandwidth = 0.5 * bandwidth / tf.log(tf.cast(n_x, tf.float32) + 1)
        bandwidth = tf.maximum(tf.stop_gradient(bandwidth), 1e-5)
    Kxy = tf.exp(-pairwise_dists / bandwidth / 2)
    return Kxy


def _squeeze(tensors, n_particles):
    return tf.concat(
        [tf.reshape(t, [n_particles, -1]) for t in tensors], axis=1)


def _unsqueeze(squeezed, original_tensors):
    ret = []
    offset = 0
    for t in original_tensors:
        size = tf.reduce_prod(tf.shape(t)[1:])
        buf = squeezed[:, offset: offset+size]
        offset += size
        ret.append(tf.reshape(buf, tf.shape(t)))
    return ret 


def stein_variational_gradient(
    log_joint, observed, latent, kernel=None):
    n_particles = None
    kernel = kernel or rbf_kernel
    assert_ops = []
    for param, value_tensor in latent.items():
        if n_particles is None:
            n_particles = tf.shape(value_tensor)[0]
        else:
            assert_ops.append(
                tf.assert_equal(n_particles, tf.shape(value_tensor)[0]))

    with tf.control_dependencies(assert_ops):
        observed = observed.copy()
        observed.update(latent)
        log_lhood = log_joint(observed)
        params = [v for _, v in latent.items()]
        params_squeezed = _squeeze(params, n_particles)
        Kxy = kernel(params_squeezed, tf.stop_gradient(params_squeezed))
        # We want dxkxy[x] := -sum_y\frac{\partial K(x,y)}{\partial y}
        # tf does not support Jacobian, and tf.gradients(Kxy, theta) returns
        # ret[x] = \sum_y\frac{\partial K(x,y)}{\partial x}
        # For stationary kernel ret = -dxkxy. TODO: non-stationary kernel
        dxkxy = -tf.gradients(Kxy, params_squeezed)[0]
        grads = _squeeze(tf.gradients(log_lhood, params), n_particles)
        new_grads = (tf.matmul(Kxy, grads) + dxkxy) / tf.cast(
            n_particles, tf.float32)
    return list(zip(_unsqueeze(new_grads, params), params))

