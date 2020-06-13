import glob
import random

import numpy as np
import tensorflow as tf
from scipy.misc import imsave as imsave
from skimage import img_as_float
from skimage.color import rgb2yuv
from skimage.io import imread
import os
import opt
from imresize import imresize
from psnr2 import psnr

scale_parameter = 4
high_dim = 72
low_dim = high_dim // scale_parameter

learning_rate = 0.001
training_iters = 50000
batch_size = 64
display_step = 250

xh = tf.placeholder("float", [batch_size, high_dim, high_dim, 1])
xb = tf.placeholder("float", [batch_size, high_dim, high_dim, 1])

btrain = tf.placeholder(tf.bool)


def load_train_data(dirpath):
    jpgfiles = glob.glob(dirpath + '/*.jpg')

    vimagedata = []
    for idx, jpgf in enumerate(jpgfiles):

        data = img_as_float(imread(jpgf))
        shape = np.shape(data)

        if len(shape) != 3:
            continue

        vimagedata.append(jpgf)

    return vimagedata


def randint(mx, mn=0):
    return random.randint(mn, mx)


def get_data(vimagedata):
    from sklearn.feature_extraction.image import extract_patches
    vdataname = vimagedata.pop()
    data = img_as_float(imread(vdataname))
    data = rgb2yuv(data)[:, :, 0:1]

    data_ik = extract_patches(data, (high_dim, high_dim, 1), (high_dim // 2, high_dim // 2, 1)).reshape(
        [-1] + list((high_dim, high_dim, 1)))

    return data_ik


def next_batch(vimagedata, scalar_scale_param=scale_parameter):
    data_ik = get_data(vimagedata)
    count = 0
    batch_data = None
    while True:
        vdatal = []
        vdatah = []
        vdatab = []
        if np.shape(data_ik)[0] > batch_size:
            batch_data, data_ik = data_ik[0:batch_size, :, :], data_ik[batch_size:, :, :]
        elif np.shape(data_ik)[0] == batch_size:
            batch_data = data_ik
            data_ik = get_data(vimagedata)
        else:
            data_ik = np.concatenate((data_ik, get_data(vimagedata)), axis=0)
            continue
        for i in range(batch_size):
            highres = batch_data[i, :, :]

            lowres = imresize(highres, scalar_scale=1.0 / scalar_scale_param)
            bicubic = imresize(lowres, scalar_scale=scalar_scale_param)

            vdatal.append(lowres)
            vdatah.append(highres)
            vdatab.append(bicubic)

        yield np.asarray(vdatah), np.asarray(vdatal), np.asarray(vdatab)


def binary_srcnn_network_w_binarize_act(x, name='srcnn', reuse=False, sr_factor=scale_parameter):

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        h1 = opt.conv2d(x, 64, kernel=3, stride=1, name='conv2d_0')
        h2 = opt.lrelu(h1, name='pr_0')

        h1 = opt.binary_conv2d(h2, 64, kernel=3, stride=1, name='conv2d_11')
        h1 = opt.binarize_act(h1, name='pr_11')

        h1 = opt.binary_conv2d(h1, 64, kernel=3, stride=1, name='conv2d_12')
        h2 = opt.binarize_act(h1, name='pr_12') + h2

        h1 = opt.binary_conv2d(h2, 64, kernel=3, stride=1, name='conv2d_21')
        h1 = opt.binarize_act(h1, name='pr_21')

        h1 = opt.binary_conv2d(h1, 64, kernel=3, stride=1, name='conv2d_22')
        h2 = opt.binarize_act(h1, name='pr_22') + h2

        h3 = opt.conv2d(h2, 1, kernel=3, stride=1, name='sub2')

    return h3



def pixel_loss(x):
    val = x ** 2
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(val, axis=[1, 2, 3])))

#####################

yh = binary_srcnn_network_w_binarize_act(xb)
cost_l2 = pixel_loss(yh - xh)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_l2)

#####################

saver = tf.train.Saver()


def main_sr():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        step = 1

        vdata = load_train_data('/media/mmrg/DATA1/ikb/inci_kadir/yenisrkod/val2017')
        generated = next_batch(vdata)
        import time
        lasttime = time.time()
        while step <= training_iters:
            if len(vdata) == 0:
                vdata = load_train_data('/media/mmrg/DATA1/ikb/inci_kadir/yenisrkod/val2017')
                generated = next_batch(vdata)
            batch_xh, _, batch_xb = next(generated)
            _, loss = sess.run([opt, cost_l2], feed_dict={xb: batch_xb, xh: batch_xh, btrain: True})

            if step % display_step == 0:
                print("Iter " + str(step) + ", Minibatch CL= ", loss)
                ypr, ypp = sess.run([yh, yh], feed_dict={xb: batch_xb, xh: batch_xh, btrain: False})

                vpp = []
                vbp = []
                for idx, pr in enumerate(ypr):
                    vpp.append(psnr(np.float32(batch_xh[idx]), np.float32(pr), 1.))
                    vbp.append(psnr(np.float32(batch_xh[idx]), np.float32(batch_xb[idx]), 1.))

                    imsave('test/img' + str(idx) + '_p.png', np.reshape(pr, (high_dim, high_dim)))
                    imsave('test/img' + str(idx) + '_o.png', np.reshape(batch_xh[idx], (high_dim, high_dim)))
                    imsave('test/img' + str(idx) + '_b.png', np.reshape(batch_xb[idx], (high_dim, high_dim)))

                print('PSNR: M:', np.mean(vpp), ' PSNR: B:', np.mean(vbp))
            if step % 5000 == 0:
                path = 'MMI713/binary_srcnn_new' + str(scale_parameter) + 'x/' + str(step)
                os.makedirs(path)
                saver.save(sess, 'MMI713/binary_srcnn_new' + str(scale_parameter) + 'x/' + str(step) + '/model.ckpt')
                print("time: " + str(time.time() - lasttime))
                lasttime = time.time()

            step += 1


main_sr()
