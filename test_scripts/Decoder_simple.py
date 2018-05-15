import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

tf.reset_default_graph()

# Inputs and labels
inputs = tf.placeholder(tf.float32, shape = [None, 4, 4, 4, 1])
labels = tf.placeholder(tf.float32, shape = [None, 32, 32, 32, 2]) #labels should be in one-hot form (2 classes)
# labels = tf.placeholder(tf.float32, shape = [None, 10])

# Define the number of filters in each conv layer
# n_deconvfilter = [32, 32, 32, 16, 8, 2]
n_deconvfilter = [128, 128, 128, 64, 32, 2]
initializer = tf.variance_scaling_initializer(scale=2.0)


# Define the model structure
unpool1 = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(inputs)
# unpool1 = tf.layers.conv3d_transpose( inputs=inputs, filters = 1 , kernel_size=(2,2,2), strides = 2, padding="same")
conv1 = tf.layers.conv3d( inputs = unpool1, filters = n_deconvfilter[1], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu)

unpool2 = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(conv1)
# unpool2 = tf.layers.conv3d_transpose( inputs=conv1, filters = n_deconvfilter[1] , kernel_size=2)
conv2 = tf.layers.conv3d( inputs = unpool2, filters = n_deconvfilter[2], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

unpool3 = tf.keras.layers.UpSampling3D( size = [2, 2, 2] )(conv2)
# unpool3 = tf.layers.conv3d_transpose( inputs=conv2, filters = n_deconvfilter[2] , kernel_size=2)
conv3 = tf.layers.conv3d( inputs = unpool3, filters = n_deconvfilter[3], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

conv4 = tf.layers.conv3d( inputs = conv3, filters = n_deconvfilter[4], kernel_size = 3, padding="same", activation = tf.nn.leaky_relu )

logits = tf.layers.conv3d( inputs = conv4, filters = n_deconvfilter[5], kernel_size = 3, padding="same")

outputs = tf.contrib.layers.softmax(logits)


loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels,
    logits=logits,
)
loss = tf.layers.flatten(loss)
loss = tf.reduce_mean(loss);

predictions = tf.argmax(outputs, axis = 4)


# optimizer = tf.train.AdamOptimizer(1e-3)
optimizer = tf.train.AdamOptimizer(7e-4)
optimizer = optimizer.minimize(loss)

# Test
# random.seed(1)
batch_size = 1
X_train = np.ones((batch_size,4,4,4,1))
y_train_0 =  np.random.choice([0,1], size = (batch_size, 32, 32, 32, 1))

# Change y_train_0 into one-hot expression
y_train_flat = y_train_0.reshape(-1)
y_train_onehot = np.zeros((batch_size*32*32*32, 2))
y_train_onehot[np.arange(batch_size*32*32*32), y_train_flat] = 1
y_train = y_train_onehot.reshape((batch_size, 32, 32, 32, 2))
# y_train =  np.random.choice([0,1], size = (batch_size, 10))

print('Training data shape: ', X_train.shape)
print('Training label shape: ', y_train.shape)

# Train the model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# batch_size = 128;
num_batches = int(len(y_train)/batch_size)
epochs = 100
print('The number of batches is: ', num_batches)

for i in range(epochs):
    for j in range(num_batches):
        X_batch = X_train[j*batch_size:(j+1)*batch_size, :, :, :, :]
        y_batch = y_train[j*batch_size:(j+1)*batch_size, :, :, :, :]
        # y_batch_flat = y_train[j*batch_size:(j+1)*batch_size, :]
        sess.run(optimizer, feed_dict = {inputs: X_train, labels: y_train})

    if i%5 == 0:
        results = sess.run([outputs, loss, predictions], feed_dict = {inputs: X_train, labels: y_train})
        # test = sess.run(loss, feed_dict = {inputs: X_train, labels: y_train})
        object = results[0]
        loss_test = results[1]
        prediction = results[2]
        accuracy = np.sum( prediction.reshape(y_train_0.shape) == y_train_0 ) / y_train_0.size
        print('================================')
        print('epoch: ', i)
        print(prediction.shape)
        print(object.shape)
        # print('object: ', object)
        print('loss:', loss_test)
        print('accuracy: ', accuracy)

sess.close()
