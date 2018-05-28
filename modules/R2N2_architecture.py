import tensorflow as tf
from modules import encoder
from modules import decoder
from modules import simple_lstm


def R2N2_model(image_placeholder, label_placeholder, mini_batch_size, H,W,C, NX, NY, NZ, T, learning_rate = 5e-4):
    '''
    :param mini_batch_size:
    :param H:
    :param W:
    :param C:
    :param NX:
    :param NY:
    :param NZ:
    :param T:
    :param learning_rate:
    :return:
    '''
    # # sequence_placeholder = tf.placeholder(tf.float32, shape = [batch_size,T,H,W,C])
    # image_placeholder = tf.placeholder(tf.float32, shape=[mini_batch_size, H, W, C])
    # # need the 2 because of one-hot encoding for softmax
    # label_placeholder = tf.placeholder(tf.float32, shape=[mini_batch_size, NX, NY, NZ, 2]);

    ## specify total graph
    sequence = [image_placeholder for i in range(T)]
    encoded_sequence = list();
    for image in sequence:
        encoded_out = encoder.encoder(image);
        encoded_sequence.append(encoded_out);

    # convert encoded sequence, which is a list of tensors
    # to a tensor of tensors
    encoded_sequence = tf.stack(encoded_sequence)
    encoded_sequence = tf.transpose(encoded_sequence, perm=[1, 0, 2])

    ## after we use the encoder, we should have a sequence of dense outputs
    conv_lstm_hidden = simple_lstm.simple_lstm(encoded_sequence);


    ##decode the lstm output, which is just the hidden state, 3D
    # pass it through the decoder
    n_deconvfilter = [128, 128, 128, 64, 32, 2]
    logits = decoder.decoder(conv_lstm_hidden, n_deconvfilter)

    # squeeze logits so it is 4 dimensional
    logits = tf.squeeze(logits);
    # check shape

    outputs = tf.contrib.layers.softmax(logits)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=label_placeholder,
        logits=logits,
    )
    loss = tf.layers.flatten(loss)
    loss = tf.reduce_mean(loss);

    ## define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss)

    #
    predictions = tf.argmax(outputs, axis=4)

    return loss, optimizer, predictions