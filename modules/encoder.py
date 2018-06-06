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

def res_block(input_tensor, kernel_size, filters):
    '''
    resnet is simply a number of these blocks stacked togther
    #for the shortcut, we need to add in a 1x1 conv in situations
    # where input and output shapes are not equal (by shape, we mean num_channels)
    # this is actually a 'BOTTLE-NECK BLOCK... designed to trim out feature maps before feeding them into the conv op

    :param input
    :return:
    '''
    filters1, filters2, filters3 = filters
    shortcut = input_tensor ## preserve the input
    #1d conv so we can scale the num_channels
    x = tf.layers.conv2d(input_tensor, filters = filters1, kernel_size = (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    #apply the conv
    x = tf.layers.Conv2D(filters2, kernel_size,
               padding='same')(x)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    #1d conv so we can project back to the original conv size
    x = tf.layers.conv2d(x, filters = filters3, kernel_size =(1, 1))
    x = tf.layers.batch_normalization(x)

    x = tf.add(x, input_tensor); #technically, this is down-sampled
    x = tf.nn.relu(x);
    return x

def res_block_1(inputs, filters, strides, training = True):
  shortcut = inputs

  inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=3, strides=strides, padding = 'same')
  inputs = tf.layers.batch_normalization(inputs, training)
  inputs = tf.nn.relu(inputs)

  inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=3, strides=1, padding = 'same')
  inputs = tf.layers.batch_normalization(inputs, training)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs

def encoder_resnet_sequence(input_batch_list, filter_array =[16, 16, 16] ):
    encoded_list = list();
    for input_batch in input_batch_list:
        conv7 = tf.layers.conv2d(inputs=input_batch, filters=filter_array[0], kernel_size=(2, 2))
        # use a for loop for the remaining 5 3x3 convs
        inputs = conv7;
        kernel_size = (2,2)


        ## first downsampling conv layer
        inputs = tf.layers.batch_normalization(inputs);
        inputs = tf.nn.relu(inputs);

        ## define first resblock
        for i in range(2):
            inputs = res_block_1(inputs, filter_array[0], strides = 1)
        kernel_size = (2,2);

        ## second downsampling conv layer
        inputs = tf.layers.conv2d(inputs, filters=filter_array[1], kernel_size=(3,3))
        inputs = tf.layers.batch_normalization(inputs);
        inputs = tf.nn.relu(inputs);
        inputs = tf.layers.max_pooling2d(inputs, pool_size=(2,2), strides=(2, 2))

        ## define second resblock
        for i in range(3):
            conv_out = res_block_1(inputs, filter_array[1], strides = 1)

        ## third downsampling conv layer
        inputs = tf.layers.conv2d(conv_out, filters=filter_array[2], kernel_size=(2,2))
        inputs = tf.layers.batch_normalization(inputs);
        inputs = tf.nn.relu(inputs);
        inputs = tf.layers.max_pooling2d(inputs, pool_size=(2,2), strides=(2, 2))

        ## define second resblock
        for i in range(3):
            conv_out = res_block_1(inputs, filter_array[2], strides = 1)


        batch_norm = tf.layers.batch_normalization(conv_out)
        dropout = tf.layers.dropout(batch_norm, rate=0.4);  # rate is the drop rate
        pool3 = tf.layers.max_pooling2d(inputs=dropout, pool_size=[2, 2], strides=2)
        pool7 = pool3;
        # add in dense layer
        pool_flat = tf.contrib.layers.flatten(pool7)
        dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
        encoded_list.append(dense);
    # run this
    return encoded_list;


def encoder_resnet1(input_batch):
    conv7 = tf.layers.conv2d(inputs=input_batch, filters=64, kernel_size=(2, 2))
    # use a for loop for the remaining 5 3x3 convs
    inputs = conv7;
    kernel_size = (2,2)
    filter_array = [64,64,32]

    ## first downsampling conv layer
    inputs = tf.layers.batch_normalization(inputs);
    inputs = tf.nn.relu(inputs);

    ## define first resblock
    for i in range(2):
        inputs = res_block_1(inputs, filter_array[0], strides = 1)
    kernel_size = (2,2);

    ## second downsampling conv layer
    inputs = tf.layers.conv2d(inputs, filters=64, kernel_size=(3,3))
    inputs = tf.layers.batch_normalization(inputs);
    inputs = tf.nn.relu(inputs);
    inputs = tf.layers.max_pooling2d(inputs, pool_size=(2,2), strides=(2, 2))

    ## define second resblock
    for i in range(3):
        conv_out = res_block_1(inputs, filter_array[1], strides = 1)

    ## third downsampling conv layer
    inputs = tf.layers.conv2d(conv_out, filters=32, kernel_size=(2,2))
    inputs = tf.layers.batch_normalization(inputs);
    inputs = tf.nn.relu(inputs);
    inputs = tf.layers.max_pooling2d(inputs, pool_size=(2,2), strides=(2, 2))

    ## define second resblock
    for i in range(3):
        conv_out = res_block_1(inputs, filter_array[2], strides = 1)


    batch_norm = tf.layers.batch_normalization(conv_out)
    dropout = tf.layers.dropout(batch_norm, rate=0.4);  # rate is the drop rate
    pool3 = tf.layers.max_pooling2d(inputs=dropout, pool_size=[2, 2], strides=2)
    pool7 = pool3;
    # add in dense layer
    pool_flat = tf.contrib.layers.flatten(pool7)
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

    # run this
    return dense;

def encoder_resnet2(input_batch, start_filter_array = [16,16, 16], end_filter_array = [8,8,4]):
    conv7 = tf.layers.conv2d(inputs=input_batch, filters=end_filter_array[0], kernel_size=(2, 2))
    # use a for loop for the remaining 5 3x3 convs
    inputs = conv7;
    kernel_size = (4,4)
    filter_array = [start_filter_array[0], start_filter_array[0],end_filter_array[0]]

    ## first downsampling conv layer
    inputs = tf.layers.batch_normalization(inputs);
    inputs = tf.nn.relu(inputs);

    ## define first resblock
    for i in range(2):
        inputs = res_block(inputs, kernel_size, filters = filter_array)
    kernel_size = (4,4);

    ## second downsampling conv layer
    inputs = tf.layers.conv2d(inputs, filters=end_filter_array[1], kernel_size=kernel_size)
    inputs = tf.layers.batch_normalization(inputs);
    inputs = tf.nn.relu(inputs);
    inputs = tf.layers.max_pooling2d(inputs, pool_size=(2,2), strides=(2, 2))

    ## define second resblock
    filter_array2 = [start_filter_array[1], start_filter_array[1],end_filter_array[1]]
    for i in range(2):
        conv_out = res_block(inputs, kernel_size, filters = filter_array2)

    # ## third downsampling conv layer
    # inputs = tf.layers.conv2d(conv_out, filters=end_filter_array[2], kernel_size=(2,2))
    # inputs = tf.layers.batch_normalization(inputs);
    # inputs = tf.nn.relu(inputs);
    # inputs = tf.layers.max_pooling2d(inputs, pool_size=(2,2), strides=(2, 2))
    #
    # ## define second resblock
    # filter_array3 = [start_filter_array[2], start_filter_array[2],end_filter_array[2]]
    #
    # for i in range(3):
    #     conv_out = res_block(inputs, kernel_size, filters = filter_array3)
    #

    batch_norm = tf.layers.batch_normalization(conv_out)
    dropout = tf.layers.dropout(batch_norm, rate=0.4);  # rate is the drop rate
    pool3 = tf.layers.max_pooling2d(inputs=dropout, pool_size=[2, 2], strides=2)
    pool7 = pool3;
    # add in dense layer
    pool_flat = tf.contrib.layers.flatten(pool7)
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

    # run this
    return dense;