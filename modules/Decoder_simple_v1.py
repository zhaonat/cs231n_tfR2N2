import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# Define the model structure
def decoder_simple( inputs, in_dim, batch_size, n_deconvfilter):
    # n_deconvfilter = [4, 4, 4, 4, 4, 2]
    initializer = tf.variance_scaling_initializer(scale=2.0)

    # Define the model structure
    # X = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(inputs) #0.71, 0.70
    X = tf.layers.conv3d_transpose( inputs=inputs, filters = 2 , kernel_size=(2,2,2), strides = 2, padding="same") #0.75
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[1], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu)

    # X = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(X)
    X = tf.layers.conv3d_transpose( inputs=X, filters = 2 , kernel_size=(2,2,2), strides = 2, padding="same")
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[2], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    # X = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(X)
    X = tf.layers.conv3d_transpose( inputs=X, filters = 2 , kernel_size=(2,2,2), strides = 2, padding="same")
    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[3], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    X = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

    logits = tf.layers.conv3d( inputs = X, filters = n_deconvfilter[5], kernel_size = 3, padding="same")

    outputs = tf.contrib.layers.softmax(logits)

    return [logits, outputs]


# Calculate the loss
def decoder_loss( logits, labels, out_dim, batch_size ):
    labels_reshape = tf.reshape(labels, [-1, 2])
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels,
        logits=logits,
    )
    loss = tf.reduce_mean(loss)
    return loss


# Predict 3D objects in voxels
def decoder_predict( outputs ):
    predictions = tf.argmax(outputs, axis = 4)
    return predictions


# Define optimizer
def decoder_optimize( loss, learning_rate ):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss)
    return optimizer

# Turn the labels to one-hot expression
def go_one_hot( y, batch_size, out_dim ):
    y_flat = y.reshape(-1)
    y_onehot = np.zeros((batch_size * out_dim * out_dim * out_dim, 2))
    y_onehot[np.arange(batch_size * out_dim * out_dim * out_dim), y_flat] = 1
    y_onehot = y_onehot.reshape((batch_size, out_dim, out_dim, out_dim, 2))
    return y_onehot


# print and record performance
def print_performance(i, results, loss_history, acc_history, i_history, y):
    object = results[0]
    loss_test = results[1]
    prediction = results[2]
    accuracy = np.sum( prediction.reshape(y.shape) == y) / y.size
    loss_history.append(loss_test)
    acc_history.append(accuracy)
    i_history.append(i)
    print('================================')
    print('epoch: ', i)
    print('loss:', loss_test)
    print('accuracy: ', accuracy)
    return [loss_history, acc_history, i_history]


# Train the decoder
def train_decoder():
    in_dim = 4
    out_dim = 32
    batch_size = 1
    learning_rate = 5e-4
    # Define the number of filters in each conv layer
    n_deconvfilter = [32, 32, 32, 16, 8, 2]

    # Training data
    X_train = np.random.rand(batch_size, in_dim, in_dim, in_dim, 2)
    y_train =  np.random.choice([0,1], size = (batch_size, out_dim, out_dim, out_dim, 1))
    y_train_onehot = go_one_hot( y_train, batch_size, out_dim )

    print('Training data shape: ', X_train.shape)
    print('Training label shape: ', y_train.shape)

    # Define the model
    tf.reset_default_graph()

    inputs = tf.placeholder(tf.float32, shape = [batch_size, in_dim, in_dim, in_dim, 2])
     #labels should be in one-hot form (2 classes)
    labels = tf.placeholder(tf.float32, shape = [batch_size, out_dim, out_dim, out_dim, 2])

    [logits, outputs] = decoder_simple( inputs, in_dim, batch_size, n_deconvfilter )
    loss = decoder_loss( logits, labels, out_dim, batch_size )
    predictions = decoder_predict( outputs )
    optimizer = decoder_optimize( loss,learning_rate )

    # Train the model
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num_batches = int(len(y_train)/batch_size)
    print('The number of batches is: ', num_batches)

    epochs = 2000 # accuracy: 80.57%
    loss_history = []
    acc_history = []
    i_history = []

    for i in range(epochs):
        if i%5 == 0:
            results = sess.run([outputs, loss, predictions], feed_dict = {inputs: X_train, labels: y_train_onehot})
            [loss_history, acc_history, i_history] = print_performance(i, results, loss_history, acc_history, i_history, y_train)

        for j in range(num_batches):
            X_batch = X_train[j*batch_size:(j+1)*batch_size, :, :, :, :]
            y_batch = y_train[j*batch_size:(j+1)*batch_size, :, :, :, :]
            sess.run(optimizer, feed_dict = {inputs: X_train, labels: y_train_onehot})

    sess.close()

    plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(i_history, loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    ax2 = plt.subplot(122)
    ax2.plot(i_history, acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()

# Run the training process
train_decoder()
