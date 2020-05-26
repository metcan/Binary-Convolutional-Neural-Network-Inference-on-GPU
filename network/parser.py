import tensorflow as tf
import os
import numpy as np


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        pass


def comp_params(x):
    alpha = tf.reduce_mean(tf.abs(x), reduction_indices=[0, 1, 2], keep_dims=False, name=None)
    weight = tf.sign(x)
    return weight, alpha


path = './weights/'
create_dir(path)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()

    # write tensor names into a file

    trainable_variables = tf.get_default_graph().get_collection("trainable_variables")
    # read tensor names from file
    # calculate alpha and binary weight value except for first and last layer
    # write them into a file with their names
    for v in trainable_variables:
        line = v.name
        line = line.split('/')
        name = line[-2] + '_' + line[-1]

        if line[-1] == 'weight:0':
            weight, alpha = comp_params(v)
            weight = (weight.eval())
            alpha = alpha.eval()
            with open(path + name + '.txt', 'w') as p:
                p.write(str(weight.shape) + '\n')
                for i in range(weight.shape[0]):
                    p.write('\n')
                    for j in range(weight.shape[1]):
                        p.write('\n')
                        for k in range(weight.shape[2]):
                            p.write('\n')
                            for l in range(weight.shape[3]):
                                p.write(str(int(weight[i][j][k][l])) + ',')

            with open(path + name + '_alpha.txt', 'w') as q:
                q.write(str(alpha.shape) + '\n')
                for i in range(alpha.shape[0]):
                    q.write(str(alpha[i]) + ',')

        if line[-1] == 'bias:0':
            bias = v.eval()
            with open(path + name + '.txt', 'w') as p:
                p.write(str(bias.shape) + '\n')
                for i in range(bias.shape[0]):
                    p.write(str(bias[i]) + ',')

    a = np.loadtxt
