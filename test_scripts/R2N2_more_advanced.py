import tensorflow as tf
import numpy as np
from modules import encoder
from modules import decoder_v1
from modules import simple_lstm
import settings
import os
import pickle
from data_processing import pickle_to_data

from data_processing import subsample_voxel
data_dir = os.path.join(settings.ROOT_DIR, 'data_sample');
f = open(os.path.join(data_dir, 'R2N2_9_batch_3.p'), 'rb');

batch_sample = pickle.load(f);
print(batch_sample.keys())
X, y = pickle_to_data.unravel_batch_pickle(batch_sample)
y0 = y;
X_nparr = np.array(X);
batch_size,T,H,W,C = X_nparr.shape;

sample_image = X_nparr[0,0,:,:,:]; print(sample_image.shape)

## permute; exchange batch and time size
X_permute = np.transpose(X_nparr, axes = [1,0,2,3,4])

## convert this back to a list of arrs
X_final = [X_permute[t, :,:,:,:] for t in range(T)];

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

#sequence_placeholder = tf.placeholder(tf.float32, shape = [batch_size,T,H,W,C])
sequence_placeholder = tf.placeholder(tf.float32, shape = [T,batch_size,H,W,C])
image_placeholder = tf.placeholder(tf.float32, shape = [batch_size,H,W,C])
label_placeholder = tf.placeholder(tf.float32, shape = [batch_size, NX,NY,NZ]);

## specify total graph
sequence = [image_placeholder for i in range(T)]
encoded_sequence = list();
for image in sequence:
    encoded_out = encoder.encoder(image);
    encoded_sequence.append(encoded_out);

#check type of encoded sequence
print(type(encoded_sequence));
print(encoded_sequence[0].shape)

#convert encoded sequence, which is a list of tensors
#to a tensor of tensors
encoded_sequence = tf.stack(encoded_sequence)
encoded_sequence = tf.transpose(encoded_sequence, perm = [1,0,2])
print(encoded_sequence.shape)

## after we use the encoder, we should have a sequence of dense outputs
conv_lstm_hidden = simple_lstm.simple_lstm(encoded_sequence);

print('lstm hidden shape:')
print(conv_lstm_hidden.shape)

##decode the lstm output, which is just the hidden state, 3D
# pass it through the decoder
n_deconvfilter = [128, 128, 128, 64, 32, 2]
logits = decoder_v1.decoder_simple(conv_lstm_hidden, n_deconvfilter)

#squeeze logits so it is 4 dimensional
logits = tf.squeeze(logits);
#check shape
print('final output shape:')
print(logits.shape)

outputs = tf.contrib.layers.softmax(logits)
print(outputs.shape)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y_one_hot,
    logits=logits,
)
loss = tf.layers.flatten(loss)
loss = tf.reduce_mean(loss);

## define optimizer
optimizer = tf.train.AdamOptimizer(1e-4)
optimizer = optimizer.minimize(loss)

#
predictions = tf.argmax(outputs, axis = 4)
print(predictions.shape)

## =================== RUN TENSORFLOW SESSION ==================##
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
epochs = 10000;
loss_history = list(); accuracy = list()
for epoch in range(epochs):

    sess.run(optimizer, feed_dict = {i: d for i, d in zip(sequence, X_final)})
    loss_epoch = sess.run(loss, feed_dict = {i: d for i, d in zip(sequence, X_final)})

    prediction = sess.run(predictions, feed_dict = {i: d for i, d in zip(sequence, X_final)})

    accuracy.append(np.mean(prediction == y))
    if(epoch%200 == 0):
        print('epoch: '+str(epoch)+' loss: '+str(loss_epoch))
        print(prediction.shape)
        print(np.mean(prediction == y))
    loss_history.append(loss_epoch);
    #predictions = tf.argmax(outputs, axis = 4)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

for j in range(batch_size):
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
plt.show()