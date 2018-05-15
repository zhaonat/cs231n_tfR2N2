import tensorflow as tf
import numpy as np

# Inputs: sequences of 1024-entry vectors
# Input shape is [batch_size, sequence_length, encoded_size]
#
# Outputs: 3D-tensor that can feed into the decoder
# Output shape = [batch_size, 4, 4, 4, Nh]
# where Nh is the hidden state size
def simple_lstm(inputs, Nh=4, initializer=None):
    num_hidden = 4**3 * Nh

    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, initializer=initializer, state_is_tuple=True)
    hidden_state, cell_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    hidden_state = tf.transpose(hidden_state, [1, 0, 2])
    outputs = hidden_state[-1]
    outputs = tf.reshape(outputs, [-1, 4, 4 ,4, Nh])
    return outputs

# Train the LSTM above on a simple task of counting the number of 1s
# in a given sequence, and outputs if the number is even or odd
def train_simple_lstm(seq_length=5, encoded_size=1):
    # Data
    train_num = 10000
    X_train = np.random.randint(2, size=[train_num, seq_length, encoded_size])
    y_train = np.sum(X_train, axis=(1,2)) % 2
    y_train = y_train.reshape([-1, 1])

    test_num = 100
    X_test = np.random.randint(2, size=[test_num, seq_length, encoded_size])
    y_test = np.sum(X_test, axis=(1,2)) % 2
    y_test = y_test.reshape([-1, 1])

    # Place holders
    inputs = tf.placeholder(tf.float32, shape=[None, seq_length, encoded_size])
    labels = tf.placeholder(tf.float32, shape=[None, 1])

    initializer = tf.variance_scaling_initializer(scale=2.0)

    # Network structure
    X = simple_lstm(inputs)
    X = tf.layers.flatten(X)
    # X = tf.layers.dense(X, 10, activation=tf.nn.relu, kernel_initializer=initializer)
    outputs = tf.layers.dense(X, 1, kernel_initializer=initializer)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01) # lr = 0.05 for Nh = 2
    optimizer = optimizer.minimize(loss)

    # Initialize and run the graph
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 100
    for epoch_index in range(20):
        for batch_index in range(train_num // batch_size):
            X_train_batch = X_train[batch_index*batch_size : (batch_index+1)*batch_size]
            y_train_batch = y_train[batch_index*batch_size : (batch_index+1)*batch_size]

            results = sess.run(optimizer, feed_dict={inputs: X_train_batch, labels: y_train_batch})
            if(batch_index % 10 == 0):
                results_train_batch = sess.run([outputs, loss], feed_dict={inputs: X_train_batch, labels: y_train_batch})
                results_test = sess.run([outputs, loss], feed_dict={inputs: X_test, labels: y_test})
                print('------------------------------')
                print('Epoch %d. Batch %d.' % (epoch_index, batch_index))
                print('Train accuracy: %f. Loss: %f.' % (np.mean((results_train_batch[0]>0) == y_train_batch), results_train_batch[1]))
                print('Test accuracy: %f. Loss: %f.' % (np.mean((results_test[0]>0) == y_test), results_test[1]))

#train_simple_lstm()
