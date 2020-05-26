import tensorflow as tf


def conv2d(x, dim, kernel=5, stride=2, name='conv2d', isbias=True):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [kernel, kernel, x.get_shape()[-1], dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

        if isbias == True:
            b = tf.get_variable('bias', [dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)

    return conv


def compute_threshold(x):
    x_sum = tf.reduce_sum(tf.abs(x), reduction_indices=None, keep_dims=False, name=None)
    threshold = tf.div(x_sum, tf.cast(tf.size(x), tf.float32), name=None)
    threshold = tf.multiply(0.7, threshold, name=None)

    return threshold


def binarize_act(x, name='binary_act'):
    alpha = tf.reduce_mean(tf.abs(x), reduction_indices=None,
                           keep_dims=False, name=None)
    return alpha * binarize(x)


def compute_alpha(x):
    thresh = compute_threshold(x)
    alpha1_temp = tf.where(tf.greater(x, thresh), x, tf.zeros_like(x, tf.float32))
    alpha2_temp = tf.where(tf.less(x, -thresh), x, tf.zeros_like(x, tf.float32))

    alpha_arr = tf.add(alpha1_temp, alpha2_temp, name=None)
    alpha_arr_abs = tf.abs(alpha_arr)
    alpha_arr_abs1 = tf.where(tf.greater(alpha_arr_abs, 0), tf.ones_like(alpha_arr_abs, tf.float32),
                              tf.zeros_like(alpha_arr_abs, tf.float32))

    alpha_sum = tf.reduce_sum(alpha_arr_abs)
    n = tf.reduce_sum(alpha_arr_abs1)
    alpha = tf.div(alpha_sum, n)

    return alpha


from tensorflow.python.framework import ops


def binarize(x):
    g = tf.get_default_graph()

    with ops.name_scope('Binarized') as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x = tf.clip_by_value(x, -1, 1)
            return tf.sign(x)


def ternary_operation(x):
    g = tf.get_default_graph()

    with ops.name_scope('Tenary') as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            thresh = compute_threshold(x)
            b = tf.sign(tf.add(tf.sign(tf.add(x, thresh)), tf.sign(tf.add(x, -thresh))))

            return b


def sample_gumbel(shape, eps=1e-5):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def binary_conv2d(x, outp_dim, kernel=5, stride=2, name='conv2d', isbias=True):
    with tf.variable_scope(name):
        sz = x.get_shape().as_list()
        w = tf.get_variable('weight', [kernel, kernel, sz[3], outp_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

        wb = binarize(w)
        alpha = tf.reduce_mean(tf.abs(w), reduction_indices=None, keep_dims=False, name=None)

        wb = tf.multiply(alpha, wb)

        conv = tf.nn.conv2d(x, wb, strides=[1, stride, stride, 1], padding='SAME')

        if isbias == True:
            b = tf.get_variable('bias', [outp_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)

    return conv


def ternary_conv2d(x, outp_dim, kernel=5, stride=2, name='conv2d', isbias=True):
    with tf.variable_scope(name):
        sz = x.get_shape().as_list()
        w = tf.get_variable('weight', [kernel, kernel, sz[3], outp_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

        wb = ternary_operation(w)
        alpha = compute_alpha(w)
        wb = tf.multiply(alpha, wb)

        conv = tf.nn.conv2d(x, wb, strides=[1, stride, stride, 1], padding='SAME')

        if isbias == True:
            b = tf.get_variable('bias', [outp_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)

    return conv


def maxpool2d(x, kernel=2, stride=2, name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME')


def avgpool2d(x, kernel=2, stride=2, name='max_pool'):
    return tf.nn.avg_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME')


def batch_norm(x, epsilon=1e-5, momentum=0.99, name='batch_norm', bscale=True, training=True):
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=bscale,
                                        is_training=training, scope=name)


def lrelu(x, leak=0.1, name='lrelu'):
    return tf.maximum(x, x * leak)

def relu(x, name="relu"):
	return tf.maximum(0,x)

def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        a = tf.get_variable('param', [x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        return tf.maximum(x, x * a)



def linear(x, dim, name='linear', isbias=True):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [x.get_shape()[-1], dim], initializer=tf.contrib.layers.xavier_initializer())
        lin = tf.matmul(x, w)

        if isbias == True:
            b = tf.get_variable('bias', [dim], initializer=tf.constant_initializer(0.0))
            lin = tf.nn.bias_add(lin, b)

        return lin


def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32,
                               initializer=tf.constant_initializer(0.01))

    beta = tf.minimum(0.1, tf.maximum(beta, 0.01))

    return tf.maximum(x, beta * x)


def binary_linear(x, inp_dim, outp_dim, name='linear', isbias=True):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [inp_dim, outp_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))

        wb = binarize(w)
        alpha = tf.reduce_mean(tf.abs(w), reduction_indices=None, keep_dims=False, name=None)
        wb = wb + binarize(w - alpha * wb)
        alpha = tf.reduce_mean(tf.abs(w - alpha * wb), reduction_indices=None, keep_dims=False, name=None)
        wb = alpha * wb

        lin = tf.matmul(x, wb)

        if isbias == True:
            b = tf.get_variable('bias', [outp_dim], initializer=tf.constant_initializer(0.0))
            lin = tf.nn.bias_add(lin, b)

        return lin


def ternary_linear(x, outp_dim, name='linear', isbias=True):
    with tf.variable_scope(name):
        sz = x.get_shape().as_list()
        w = tf.get_variable('weight', [sz[1], outp_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))

        wb = ternary_operation(w)
        alpha = compute_alpha(w)
        wb = tf.multiply(alpha, wb)

        lin = tf.matmul(x, wb)

        if isbias == True:
            b = tf.get_variable('bias', [outp_dim], initializer=tf.constant_initializer(0.0))
            lin = tf.nn.bias_add(lin, b)

        return lin
