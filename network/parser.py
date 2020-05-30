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


w_path = './weights/'
b_path = './bias/'
a_path = './alpha/'
create_dir(w_path)
create_dir(b_path)
create_dir(a_path)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()

    # write tensor names into a file

    trainable_variables = tf.get_default_graph().get_collection("trainable_variables")
    # read tensor names from file
    # calculate alpha and binary weight value except for first and last layer
    # write them into a file with their names
    index = 0
    for v in trainable_variables:
        line = v.name
        line = line.split('/')
        name = line[-2] + '_' + line[-1]
        checker = line[-2]
        if line[-1] == 'weight:0':
            if checker != 'conv2d_0' and checker != 'sub2':
                weight, alpha = comp_params(v)
                weight = weight.eval()
                alpha = alpha.eval()
                flag = True

            else:
                weight = v.eval()
                flag = False

            with open(w_path + name + '.txt', 'w') as p:
                # p.write(str(weight.shape) + '\n')
                for i in range(weight.shape[3]):

                    for j in range(weight.shape[2]):

                        for k in range(weight.shape[1]):

                            for z in range(weight.shape[0]):
                                if flag:
                                    p.write(str(int(weight[z][k][j][i])) + ',')
                                else:
                                    p.write(str(weight[z][k][j][i]) + ',')
                            p.write('_')
                        p.write('/')
                    p.write('*')

            with open(w_path + 'weight_shapes' + '.txt', 'a') as p:
                p.write(name + ' : ' + str(weight.shape[3]) + ',' + str(weight.shape[2]) + ',' + str(weight.shape[1]) +
                        ',' + str(weight.shape[0]) + '\n')

            if flag:
                with open(a_path + name + '_alpha.txt', 'w') as q:
                    # q.write(str(alpha.shape) + '\n')
                    for i in range(alpha.shape[0]):
                        q.write(str(alpha[i]) + ',')

                with open(a_path + 'alpha_shapes' + '.txt', 'a') as p:
                    p.write(name + ' : ' + str(alpha.shape) + '\n')

        if line[-1] == 'bias:0':
            bias = v.eval()
            with open(b_path + name + '.txt', 'w') as p:
                # p.write(str(bias.shape) + '\n')
                for i in range(bias.shape[0]):
                    p.write(str(bias[i]) + ',')

            with open(b_path + 'bias_shapes' + '.txt', 'a') as p:
                p.write(name + ' : ' + str(bias.shape) + '\n')
