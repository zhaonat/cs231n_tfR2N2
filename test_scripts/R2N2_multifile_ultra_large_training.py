import tensorflow as tf
import numpy as np
from modules import encoder
from modules import decoder
from modules import simple_lstm
from modules import R2N2_architecture
import settings
import os
import pickle
from data_processing import intersection_over_union as iou

from data_processing import pickle_to_data
'''
this trains a model ACROSS multiple batch files...the full realization of our model
we can only store examples in up to 1000 sample batches...so when we train, we need to iterate over the files 
in the tensorflow graph
this takes a while to run
'''
from data_processing import subsample_voxel
num_files = 20;
epochs = 50;

## get all necessary sizes
data_dir = os.path.join(settings.ROOT_DIR, 'data', 'shape_net_training_test');
f = open(os.path.join(data_dir, 'R2N2_128_batch_1.p'), 'rb');

batch_sample = pickle.load(f);
print(batch_sample.keys())
sample_keys = list(batch_sample.keys())
X, y = pickle_to_data.unravel_batch_pickle(batch_sample)
y0 = y;
X_nparr = np.array(X);
batch_size, T, H, W, C = X_nparr.shape;

## perform downsampling
down_y = list();
strides = [1, 1, 1]
for i in range(len(y)):
    down_y.append(subsample_voxel.downsample(y[i], strides))
sample_model = down_y[0];
print(sample_model.shape)
[NX, NY, NZ] = sample_model.shape;


## define model artitecture
mini_batch_size = 8;
# sequence_placeholder = tf.placeholder(tf.float32, shape = [batch_size,T,H,W,C])
sequence_placeholder = tf.placeholder(tf.float32, shape=[T, mini_batch_size, H, W, C])
image_placeholder = tf.placeholder(tf.float32, shape=[mini_batch_size, H, W, C])
# need the 2 because of one-hot encoding for softmax
label_placeholder = tf.placeholder(tf.float32, shape=[mini_batch_size, NX, NY, NZ, 2]);
sequence = [image_placeholder for i in range(T)]

loss, optimizer, predictions = R2N2_architecture.R2N2_model(image_placeholder, label_placeholder, mini_batch_size, H, W,
                                                            C, NX, NY, NZ, T, learning_rate=5e-4)
print(predictions.shape)

for epoch in range(epochs):

    for train_file_index in range(1,num_files):
        print('train_file_index: '+str(train_file_index))
        data_dir = os.path.join(settings.ROOT_DIR, 'data', 'shape_net_training_test');
        f = open(os.path.join(data_dir, 'R2N2_128_batch_'+str(train_file_index)+'.p'), 'rb');

        batch_sample = pickle.load(f);
        print(batch_sample.keys())
        sample_keys = list(batch_sample.keys())
        X, y = pickle_to_data.unravel_batch_pickle(batch_sample)
        y0 = y;
        X_nparr = np.array(X);
        batch_size,T,H,W,C = X_nparr.shape;


        ## perform downsampling
        down_y = list();
        strides = [1,1,1]
        for i in range(len(y)):
            down_y.append(subsample_voxel.downsample(y[i], strides))
        sample_model = down_y[0]; print(sample_model.shape)
        [NX,NY,NZ] = sample_model.shape;

        ## convert everything to arrays
        y = np.array(down_y)
        print('voxel shape')
        print(y.shape)
        #convert to one hot
        y_one_hot = np.stack((y==0, y==1))
        y_one_hot = np.transpose(y_one_hot, axes = [1,2,3,4,0])
        # plt.figure();
        # plt.imshow(sample_image[:,:,0:3]);
        # plt.show()


        ## =================== RUN TENSORFLOW SESSION ==================##
        #Create a saver object which will save all the variables
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        loss_history = list(); accuracy = list()

        num_batches = int(batch_size/mini_batch_size);

        for i in range(num_batches):
            #need to determine X_final
            start = i*mini_batch_size; end = (i+1)*mini_batch_size;
            print(sample_keys[start:end])
            y_batch = y_one_hot[start:end];
            y_batch_flat = y[start:end]
            # print('y_batch shape')
            # print(y_batch.shape)
            X_batch = X_nparr[start:end, :, :, :, :]
            X_batch = np.transpose(X_batch, axes=[1, 0, 2, 3, 4])
            ## convert this back to a list of arrs
            X_final = [X_batch[t, :, :, :, :] for t in range(T)];

            feed_dict_input = {i: d for i, d in zip(sequence, X_final)};
            feed_dict_input[label_placeholder] = y_batch;
            #manually insert the label_placeholder


            sess.run(optimizer, feed_dict = feed_dict_input)
            loss_epoch = sess.run(loss, feed_dict = feed_dict_input)

            #reconfigure feed_dict to accept a place_holder for y
            prediction = sess.run(predictions, feed_dict = feed_dict_input)

            accuracy.append(np.mean(prediction == y_batch_flat))

        print('epoch: '+str(epoch)+' loss: '+str(loss_epoch))
        print(prediction.shape)
        print(np.mean(prediction == y_batch_flat))
        loss_history.append(loss_epoch);
            #predictions = tf.argmax(outputs, axis = 4)


## run test batch
f = open(os.path.join(data_dir, 'R2N2_128_batch_1.p'), 'rb');
batch_sample_2 = pickle.load(f);
X, y = pickle_to_data.unravel_batch_pickle(batch_sample_2)
X_nparr = np.array(X);
batch_size,T,H,W,C = X_nparr.shape;
num_batches = int(128/mini_batch_size);
test_iou = list();
print('run test case')

for i in range(num_batches-1):
    start = i * mini_batch_size;
    end = (i + 1) * mini_batch_size;

    X_batch = X_nparr[start:end, :, :, :, :]

    X_batch = np.transpose(X_batch, axes=[1, 0, 2, 3, 4])

    ## convert this back to a list of arrs
    X_final = [X_batch[t, :, :, :, :] for t in range(T)];


    ## perform downsampling
    down_y = list();
    strides = [1,1,1]
    for i in range(len(y)):
        down_y.append(subsample_voxel.downsample(y[i], strides))
    sample_model = down_y[0]; print(sample_model.shape)
    [NX,NY,NZ] = sample_model.shape;
    ## convert everything to arrays
    y = np.array(down_y)
    y_one_hot = np.stack((y==0, y==1))
    y_one_hot = np.transpose(y_one_hot, axes = [1,2,3,4,0])
    feed_dict_input = {i: d for i, d in zip(sequence, X_final)}; #size has to be fixed...
    feed_dict_input[label_placeholder] = y_one_hot[start:end];
    test_loss= sess.run(loss, feed_dict=feed_dict_input)
    test_prediction = sess.run(predictions, feed_dict=feed_dict_input)
    for k in range(mini_batch_size):
        print(iou.IoU_3D(y[k, :, :, :], test_prediction[k, :, :, :]))
        test_iou.append(iou.IoU_3D(y[k, :, :, :], test_prediction[k, :, :, :]))

print('test_loss')
print(test_loss)
print(np.mean(test_iou))

#Now, save the graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

for j in range(batch_size-1):
    fig = plt.figure(figsize = (20,8))
    ax = fig.add_subplot(1,2, 1, projection='3d')
    ax.voxels(prediction[j,:,:,:], edgecolor='k')
    ax.set(xlabel='x', ylabel='y', zlabel='z', title='prediction')

    ax = fig.add_subplot(1,2, 2, projection='3d')
    ax.voxels(y[j,:,:,:], edgecolor='k', facecolor = 'blue')
    ax.set(xlabel='x', ylabel='y', zlabel='z', title='original model')

plt.show();


plt.figure();
plt.subplot(121)
plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Sample R2N2 Preliminary Module Loss')
plt.subplot(122)
plt.plot(accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Sample R2N2 Preliminary Module Acc')
plt.tight_layout();
plt.show()