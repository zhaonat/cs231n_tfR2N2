import tensorflow as tf

def encoder(input_image):
    '''
    #input image should be a tensorflow tensor
    should an input 127x127 rgb image (W x H x C); #color channels should be last dim
    should ouput a dense representation or encoding
    '''

    conv7 = tf.layers.conv2d(
      inputs=input_image,
          filters=32,
          kernel_size=[1,1],
          padding="same",
          activation=tf.nn.relu)
    pool7 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    #use a for loop for the remaining 5 3x3 convs
    for i in range(5):
        conv3 = tf.layers.conv2d(
          inputs=pool7,
          filters=32,
          kernel_size=[3,3],
          padding="same",
          activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        pool7 = pool3;
        
    #add in dense layer
    pool_flat = tf.contrib.layers.flatten(pool7)
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
    
    #run this
    return dense;

def encoder_z(input_batch):
    conv7 = tf.layers.conv2d(inputs=input_batch, filters=32, kernel_size=(2, 2))
    # use a for loop for the remaining 5 3x3 convs
    pool7 = conv7;
    for i in range(4):
        conv3 = tf.layers.conv2d(
            inputs=pool7,
            filters=64,
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)
        batch_norm = tf.layers.batch_normalization(conv3)
        dropout = tf.layers.dropout(batch_norm, rate=0.4);  # rate is the drop rate
        pool3 = tf.layers.max_pooling2d(inputs=dropout, pool_size=[2, 2], strides=2)
        pool7 = pool3;

    # add in dense layer
    pool_flat = tf.contrib.layers.flatten(pool7)
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

    # run this
    return dense;

