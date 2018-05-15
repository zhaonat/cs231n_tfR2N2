import tensorflow as tf
import numpy as np

# Do NOT write None for batch_size

# LSTM cell class.
# This is similar to tf.nn.rnn_cell.LSTMCell with state_is_tuple=True.
# It has the potential of being modified to a 3D convolutional LSTM.
class MyLSTMCell():
    def __init__(self, num_units, num_features, params=None):
        self._num_units = num_units
        self._num_features = num_features
        if params is None:
            self._Wf = tf.Variable(tf.truncated_normal([self._num_features, self._num_units]))
            self._Wi = tf.Variable(tf.truncated_normal([self._num_features, self._num_units]))
            self._Wo = tf.Variable(tf.truncated_normal([self._num_features, self._num_units]))
            self._Wc = tf.Variable(tf.truncated_normal([self._num_features, self._num_units]))
            self._Uf = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
            self._Ui = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
            self._Uo = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
            self._Uc = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
            self._bf = tf.Variable(tf.zeros([1, self._num_units]))
            self._bi = tf.Variable(tf.zeros([1, self._num_units]))
            self._bo = tf.Variable(tf.zeros([1, self._num_units]))
            self._bc = tf.Variable(tf.zeros([1, self._num_units]))
        else:
            self._Wf = params['Wf']
            self._Wi = params['Wi']
            self._Wo = params['Wo']
            self._Wc = params['Wc']
            self._Uf = params['Uf']
            self._Ui = params['Ui']
            self._Uo = params['Uo']
            self._Uc = params['Uc']
            self._bf = params['bf']
            self._bi = params['bi']
            self._bo = params['bo']
            self._bc = params['bc']

    def __call__(self, inputs, state):
        c, h = state

        zf = tf.matmul(inputs, self._Wf) + tf.matmul(h, self._Uf) + self._bf
        f = tf.sigmoid(zf)

        zi = tf.matmul(inputs, self._Wi) + tf.matmul(h, self._Ui) + self._bi
        i = tf.sigmoid(zi)

        zo = tf.matmul(inputs, self._Wo) + tf.matmul(h, self._Uo) + self._bo
        o = tf.sigmoid(zo)

        zc = tf.matmul(inputs, self._Wc) + tf.matmul(h, self._Uc) + self._bc
        c = f * c + i * tf.tanh(zc)
        h = o * tf.tanh(c)

        state = (c, h)
        return h, state

# my_dynamic_rnn is a wrapper that executes an LSTM cell multiple times.
# This is similar to tf.nn.dynamic_rnn(...). I try to match the inputs and
# outputs to tf.nn.dynamic_rnn(...) as much as possible, but here it does not
# return the full hidden state history, as we do not need it. Instead it only
# returns the last hidden state, as well as a variable called state, which is a
# tuple of (cell_state, hidden_state).
def my_dynamic_rnn(cell, inputs):
    batch_size, time_steps, num_features = inputs.get_shape().as_list()
    num_units = cell._num_units

    h_init = tf.zeros([batch_size, num_units])
    c_init = tf.zeros([batch_size, num_units])
    state = (c_init, h_init)

    for t in range(time_steps):
        h, state = cell(inputs[:,t,:], state)
    return h, state

# my_lstm defines a specific LSTM, whose input is a batch of sequences, of
# shape [batch_size, time_steps, num_features], and output is reshaped to
# [4, 4, 4, Nh] to match the decoder input.
def my_lstm(inputs, Nh=4):
    num_hidden = 4**3 * Nh
    num_features = inputs.get_shape().as_list()[-1]

    cell = MyLSTMCell(num_hidden, num_features)
    h, state = my_dynamic_rnn(cell, inputs)

    outputs = tf.reshape(h, [-1, 4, 4 ,4, Nh])
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
def train_my_lstm():
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

    # Network structure: LSTM - Dense(1)
    X = my_lstm(inputs)
    X = tf.layers.flatten(X)
    outputs = tf.layers.dense(X, 1, kernel_initializer=initializer)

    if problem_type == 'binary_classification':
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=outputs)
        loss = tf.reduce_mean(loss)
    elif problem_type == 'regression':
        loss = tf.nn.l2_loss(truth - outputs) / batch_size

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01) # lr = 0.05 for Nh = 2
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
                print('Test accuracy: %f. Loss: %f.' % (np.mean((results_test[0]>0) == y_test), results_test[1]))

# train_my_lstm()
