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


def binarize_act(x, name='binary_act'):
    alpha = tf.reduce_mean(tf.abs(x), reduction_indices=None,
                           keep_dims=False, name=None)
    return alpha * binarize(x)


from tensorflow.python.framework import ops


def binarize(x):
    g = tf.get_default_graph()

    with ops.name_scope('Binarized') as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x = tf.clip_by_value(x, -1, 1)
            return tf.sign(x)


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

