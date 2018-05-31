import numpy as np
import random
import tensorflow as tf

def decoder_simple(inputs, n_deconvfilter):
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

    #at this point, the output shape be (16,16,16)

    unpool3 = tf.keras.layers.UpSampling3D(size=[2, 2, 2])(conv2)
    # unpool3 = tf.layers.conv3d_transpose( inputs=conv2, filters = n_deconvfilter[2] , kernel_size=2)
    conv3 = tf.layers.conv3d(inputs=unpool3, filters=n_deconvfilter[3], kernel_size=3, padding="same",
                             activation=tf.nn.leaky_relu)

    conv4 = tf.layers.conv3d(inputs=conv3, filters=n_deconvfilter[4], kernel_size=3, padding="same",
                             activation=tf.nn.leaky_relu)

    logits = tf.layers.conv3d(inputs=conv4, filters=n_deconvfilter[5], kernel_size=3, padding="same")
    return logits

# Residual decoder using con3d_transpose layers
def decoder_res_conv3dTranspose( inputs, n_deconvfilter):
    # initializer = tf.variance_scaling_initializer(scale=2.0)

    # Define the model structure
    # X = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(inputs) #0.71, 0.70
    unpool1 = tf.layers.conv3d_transpose( inputs=inputs, filters = n_deconvfilter[1] , kernel_size=(2,2,2), strides = 2, padding="same") #0.75
    X = tf.layers.conv3d( inputs = unpool1, filters = n_deconvfilter[1], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu)
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[1], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu)
    X = X + unpool1

    # X = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(X)
    unpool2 = tf.layers.conv3d_transpose( inputs=X, filters = n_deconvfilter[2] , kernel_size=(2,2,2), strides = 2, padding="same")
    X = tf.layers.conv3d( inputs = unpool2, filters = n_deconvfilter[2], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[2], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = X + unpool2

    # X = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(X)
    unpool3 = tf.layers.conv3d_transpose( inputs=X, filters=n_deconvfilter[3] , kernel_size=(2,2,2), strides = 2, padding="same")
    X = tf.layers.conv3d( inputs = unpool3, filters = n_deconvfilter[3], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[3], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    conv3 = tf.layers.conv3d( inputs = unpool3, filters = n_deconvfilter[3], kernel_size = 1, padding="same", activation = tf.nn.leaky_relu )
    X = X + conv3

    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    conv4 = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    X = X + conv4

    logits = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[5], kernel_size = 3, padding="same")

    # outputs = tf.contrib.layers.softmax(logits)

    return logits


# Residual decoder using UpSampling3D layers
def decoder_res_upsample( inputs, n_deconvfilter):
    # initializer = tf.variance_scaling_initializer(scale=2.0)

    # Define the model structure
    unpool1 = tf.keras.layers.UpSampling3D( size = [2, 2, 2])(inputs) #0.71, 0.70
    unpool1 = tf.layers.conv3d( inputs=unpool1, filters = n_deconvfilter[1], kernel_size = 1, padding="same" )
    X = tf.layers.conv3d( inputs = unpool1, filters = n_deconvfilter[1], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu)
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[1], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu)
    X = X + unpool1

    unpool2 = tf.keras.layers.UpSampling3D( size = [2, 2, 2])(X)
    unpool2 = tf.layers.conv3d( inputs=unpool2, filters = n_deconvfilter[2], kernel_size = 1, padding="same" )
    # unpool2 = tf.layers.conv3d_transpose( inputs=X, filters = 2 , kernel_size=(2,2,2), strides = 2, padding="same")
    X = tf.layers.conv3d( inputs = unpool2, filters = n_deconvfilter[2], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[2], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = X + unpool2

    unpool3 = tf.keras.layers.UpSampling3D( size = [2, 2, 2])(X)
    unpool3 = tf.layers.conv3d( inputs=unpool3, filters = n_deconvfilter[3], kernel_size = 1, padding="same" )
    # unpool3 = tf.layers.conv3d_transpose( inputs=X, filters = 2 , kernel_size=(2,2,2), strides = 2, padding="same")
    X = tf.layers.conv3d( inputs = unpool3, filters = n_deconvfilter[3], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[3], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    conv3 = tf.layers.conv3d( inputs = unpool3, filters = n_deconvfilter[3], kernel_size = 1, padding="same", activation = tf.nn.leaky_relu )
    X = X + conv3

    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    conv4 = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    X = X + conv4

    logits = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[5], kernel_size = 3, padding="same")

    # outputs = tf.contrib.layers.softmax(logits)

    return logits
