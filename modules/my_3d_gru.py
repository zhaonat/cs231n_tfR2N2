import tensorflow as tf
import numpy as np

# TL;DR
# Just use my_3d_lstm(inputs, cube_size, num_channels, filter_size)
# inputs: [batch_size, time_steps, num_features]
# outputs: [batch_size, cube_size, cube_size, cube_size, num_channels]
# Paper uses cube_size=4, num_channels=256 and filter_size=3

########################################
# Do NOT write None for batch_size

# 3D LSTM cell class.
# This is similar to tf.nn.rnn_cell.LSTMCell with state_is_tuple=True.
class My3DGRUCell():
    def __init__(self, num_features, cube_size, num_channels, filter_size, params=None):
        self.num_features = num_features
        self.cube_size = cube_size
        self.num_channels = num_channels
        self.filter_size = filter_size

        self.hidden_shape = [-1, self.cube_size, self.cube_size, self.cube_size, self.num_channels]
        self.num_units = self.cube_size**3 * self.num_channels

        self.update_dense = tf.layers.Dense(units=self.num_units, use_bias=False)
        self.reset_dense = tf.layers.Dense(units=self.num_units, use_bias=False)
        self.hidden_dense = tf.layers.Dense(units=self.num_units, use_bias=False)

        self.update_conv = tf.layers.Conv3D(filters=self.num_channels, kernel_size=self.filter_size, padding='same')
        self.reset_conv = tf.layers.Conv3D(filters=self.num_channels, kernel_size=self.filter_size, padding='same')
        self.hidden_conv = tf.layers.Conv3D(filters=self.num_channels, kernel_size=self.filter_size, padding='same')

    def __call__(self, inputs, h):

        # inputs: [batch_size, 1024] tensor from the encoder
        # h: [batch_size, self.cube_size, self.cube_size, self.cube_size, self.num_channels]
        # For simplicity, the shape of c and h is stored as self.hidden_shape
        zu_dense = self.update_dense(inputs)
        zu_conv = self.update_conv(h)
        zu = tf.reshape(zu_dense, self.hidden_shape) + zu_conv
        u = tf.sigmoid(zu)

        zr_dense = self.reset_dense(inputs)
        zr_conv = self.reset_conv(h)
        zr = tf.reshape(zr_dense, self.hidden_shape) + zr_conv
        r = tf.sigmoid(zr)

        zh_dense = self.hidden_dense(inputs)
        zh_conv = self.hidden_conv( r * h )
        zh = tf.reshape(zh_dense, self.hidden_shape) + zh_conv

        h = (1 - u) * h + u * tf.tanh(zh)

        return h

# my_dynamic_rnn is a wrapper that executes an LSTM cell multiple times.
# This is similar to tf.nn.dynamic_rnn(...). I try to match the inputs and
# outputs to tf.nn.dynamic_rnn(...) as much as possible, but here it does not
# return the full hidden state history, as we do not need it. Instead it only
# returns the last hidden state, as well as a variable called state, which is a
# tuple of (cell_state, hidden_state).
def my_dynamic_rnn(cell, inputs):
    batch_size, time_steps, num_features = inputs.get_shape().as_list()
    hidden_shape = [batch_size, *cell.hidden_shape[1:]]

    h = tf.zeros(hidden_shape)

    for t in range(time_steps):
        h = cell(inputs[:,t,:], h)
    return h

# my_3d_lstm defines a specific 3D LSTM, whose input is a batch of sequences, of
# shape [batch_size, time_steps, num_features], and output has shape
# [batch_size, cube_size, cube_size, cube_size, num_channels] to match the decoder input.
def my_3d_gru(inputs, cube_size=4, num_channels=4, filter_size=3):
    num_features = inputs.get_shape().as_list()[-1]

    cell = My3DGRUCell(num_features, cube_size, num_channels, filter_size)
    outputs = my_dynamic_rnn(cell, inputs)
    return outputs

# Creates sin sequence data. Can be used to train a regression problem.
def sin_data(data_size, time_steps=10):
    X = []
    y = []
    for i in range(data_size):
        offset = np.random.randn()
        ratio = np.random.randn()
        X.append(np.sin((np.arange(time_steps) + offset) * ratio))
        y.append(np.sin((time_steps + offset) * ratio))
    X = np.array(X)
    y = np.array(y)
    X = X.reshape([data_size, time_steps, 1])
    y = y.reshape([data_size, 1])
    return X, y

# Creates binary sequence data. Can be used to train a classification problem.
def binary_data(data_size, time_steps=10):
    X = np.random.randint(2, size=[data_size, time_steps, 1])
    y = np.sum(X, axis=(1,2)) % 2
    y = y.reshape([data_size, 1])
    return X, y

# Train the LSTM to make sure it works properly.
def train_my_gru():
    # Data
    train_size = 10000
    test_size = 100
    batch_size = 100
    time_steps = 5
    num_features = 1

    problem_type = 'regression'
    X_train, y_train = sin_data(train_size, time_steps=time_steps)
    X_test, y_test = sin_data(test_size, time_steps=time_steps)

    # problem_type = 'binary_classification'
    # X_train, y_train = binary_data(train_size, time_steps=time_steps)
    # X_test, y_test = binary_data(test_size, time_steps=time_steps)

    # Place holders. Do NOT write None for batch_size
    inputs = tf.placeholder(tf.float32, shape=[batch_size, time_steps, num_features])
    truth = tf.placeholder(tf.float32, shape=[batch_size, 1])

    initializer = tf.variance_scaling_initializer(scale=2.0)

    # Network structure: 3D LSTM - Dense(1)
    X = my_3d_gru(inputs)
    X = tf.layers.flatten(X)
    outputs = tf.layers.dense(X, 1, kernel_initializer=initializer)

    if problem_type == 'binary_classification':
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=outputs)
        loss = tf.reduce_mean(loss)
    elif problem_type == 'regression':
        loss = tf.nn.l2_loss(truth - outputs) / batch_size

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimizer = optimizer.minimize(loss)

    # Initialize and run the graph
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch_index in range(20):
        for batch_index in range(train_size // batch_size):
            X_train_batch = X_train[batch_index*batch_size : (batch_index+1)*batch_size]
            y_train_batch = y_train[batch_index*batch_size : (batch_index+1)*batch_size]

            results = sess.run(optimizer, feed_dict={inputs: X_train_batch, truth: y_train_batch})
            if(batch_index % 10 == 0):
                results_train_batch = sess.run([outputs, loss], feed_dict={inputs: X_train_batch, truth: y_train_batch})
                results_test = sess.run([outputs, loss], feed_dict={inputs: X_test, truth: y_test})
                print('------------------------------')
                print('Epoch %d. Batch %d.' % (epoch_index, batch_index))
                print('Train accuracy: %f. Loss: %f.' % (np.mean((results_train_batch[0]>0) == y_train_batch), results_train_batch[1]))
                # print('Train accuracy: %f. Loss: %f.' % (np.mean(np.abs(results_train_batch[0]-y_train_batch)), results_train_batch[1]))
                print('Test accuracy: %f. Loss: %f.' % (np.mean((results_test[0]>0) == y_test), results_test[1]))

# train_my_gru()
