import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

def decoder(inputs, n_deconvfilter):
    '''
    :param inputs: should have shape (N,X,Y,Z,C)
    :param n_deconvfilter:
    :return:
    '''
    # Define the model structure
    unpool1 = tf.keras.layers.UpSampling3D(size=[2, 2, 2])(inputs)
    # unpool1 = tf.layers.conv3d_transpose( inputs=inputs, filters = 1 , kernel_size=(2,2,2), strides = 2, padding="same")
    conv1 = tf.layers.conv3d(inputs=unpool1, filters=n_deconvfilter[1], kernel_size=3, padding="same",
                             activation=tf.nn.leaky_relu)

    unpool2 = tf.keras.layers.UpSampling3D(size=[2, 2, 2])(conv1)
    # unpool2 = tf.layers.conv3d_transpose( inputs=conv1, filters = n_deconvfilter[1] , kernel_size=2)
    conv2 = tf.layers.conv3d(inputs=unpool2, filters=n_deconvfilter[2], kernel_size=3, padding="same",
                             activation=tf.nn.leaky_relu)

    unpool3 = tf.keras.layers.UpSampling3D(size=[2, 2, 2])(conv2)
    # unpool3 = tf.layers.conv3d_transpose( inputs=conv2, filters = n_deconvfilter[2] , kernel_size=2)
    conv3 = tf.layers.conv3d(inputs=unpool3, filters=n_deconvfilter[3], kernel_size=3, padding="same",
                             activation=tf.nn.leaky_relu)

    conv4 = tf.layers.conv3d(inputs=conv3, filters=n_deconvfilter[4], kernel_size=3, padding="same",
                             activation=tf.nn.leaky_relu)

    logits = tf.layers.conv3d(inputs=conv4, filters=n_deconvfilter[5], kernel_size=3, padding="same")
    return logits


