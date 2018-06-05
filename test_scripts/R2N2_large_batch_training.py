import tensorflow as tf
import numpy as np
from modules import encoder
from modules import decoder
from modules import simple_lstm
from modules import my_3d_lstm
from modules import my_3d_gru
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import settings
import os
import pickle
from data_processing import intersection_over_union as iou

from data_processing import pickle_to_data
'''
this takes a while to run
'''
from data_processing import subsample_voxel
data_dir = os.path.join(settings.ROOT_DIR, 'data');
f = open(os.path.join(data_dir, 'planes_4045_batch', \
                      'planes_cropped_grayscale','planes_02691156_4045_batch_cropped_grayscale.p'), 'rb');
#f = open(os.path.join(data_dir, 'R2N2_128_batch_1.p'), 'rb');
class_name = 'planes';
X,y  = pickle.load(f);

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4);
## train test split
print(X_train.shape)
print(X_test.shape)

y0 = y;
X_nparr = X_train;
batch_size,T,H,W,C = X_nparr.shape;

print(y.shape)
batch_size,NX,NY,NZ = y.shape

## if you want, take a smaller subset to test
subset = 256;
X_train = X_train[0:subset,:,:,:,:];
y_train = y_train[0:subset];

batch_size,T,H,W,C = X_train.shape;

epochs = 500;
mini_batch_size = 32;

#need the 2 because of one-hot encoding for softmax
label_placeholder = tf.placeholder(tf.float32, shape = [mini_batch_size , NX,NY,NZ], name = 'voxel');

## =========================== TUNE-ABLE PARAMETERS ================================================##
sub_sequence_length = 2;
conv_lstm_hidden_channels = 24

sequence = []
for i in range(sub_sequence_length):
    sequence.append(tf.placeholder(tf.float32, shape = [mini_batch_size ,H,W,C], name = 'img_'+str(i)))

encoded_sequence = list();
print('sequence length: '+str(len(sequence)))
for image in sequence:
    #encoded_out = encoder.encoder_resnet2(image,start_filter_array = [4,4,4], end_filter_array = [8,8,4]);
    encoded_out = encoder.encoder_resnet1(image);

    encoded_sequence.append(encoded_out);

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print('total: '+str(total_parameters))

#check type of encoded sequence
print(type(encoded_sequence));
print(encoded_sequence[0].shape)

#convert encoded sequence, which is a list of tensors
#to a tensor of tensors
encoded_sequence = tf.stack(encoded_sequence)
encoded_sequence = tf.transpose(encoded_sequence, perm = [1,0,2])
print(encoded_sequence.shape)

## after we use the encoder, we should have a sequence of dense outputs
#conv_lstm_hidden = simple_lstm.simple_lstm(encoded_sequence);
conv_lstm_hidden = my_3d_gru.my_3d_gru(encoded_sequence, cube_size=4, num_channels=conv_lstm_hidden_channels, filter_size=3)


print('lstm hidden shape:')
print(conv_lstm_hidden.shape)

##decode the lstm output, which is just the hidden state, 3D
# pass it through the decoder
n_deconvfilter = [32,32,16,16,8, 1]
logits = decoder.decoder_res_conv3dTranspose(conv_lstm_hidden, n_deconvfilter)

#squeeze logits so it is 4 dimensional
logits = tf.squeeze(logits);
outputs = tf.nn.sigmoid(logits);

outputs = tf.identity(outputs, name='classification_activation');

#check shape
print('final output shape:')
print(logits.shape)

print(label_placeholder.shape)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(
#     labels=label_placeholder,
#     logits=logits,
# )

loss=tf.nn.weighted_cross_entropy_with_logits(
    targets = label_placeholder,
    logits = logits,
    pos_weight = 10,
    name=None
)

loss = tf.layers.flatten(loss)
loss = tf.reduce_mean(loss);

## define optimizer
optimizer = tf.train.AdamOptimizer(5e-5)
optimizer = optimizer.minimize(loss)



## =================== RUN TENSORFLOW SESSION ==================##
#Create a saver object which will save all the variables
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
loss_history = list(); training_accuracy = list()
num_batches = int(batch_size/mini_batch_size);
training_iou = [];
for epoch in range(epochs):

    for i in range(num_batches):
        #need to determine X_final
        start = i*mini_batch_size; end = (i+1)*mini_batch_size;
        y_batch = y[start:end]
        # print('y_batch shape')

        X_batch = X_train[start:end, :, :, :, :]
        X_batch = np.transpose(X_batch, axes=[1, 0, 2, 3, 4])
        ## convert this back to a list of arrs

        #pick a random sample of perspectives: DON'T DO THAT
        #t_sample = np.random.randint(0, T, sub_sequence_length)
        t_sample = np.arange(0, sub_sequence_length);

        X_final = [X_batch[t, :, :, :, :] for t in t_sample];

        feed_dict_input = {i: d for i, d in zip(sequence, X_final)};
        feed_dict_input[label_placeholder] = y_batch;
        #print(feed_dict_input.keys())
        #manually insert the label_placeholder
        loss_epoch, prediction, _ = sess.run([loss, outputs, optimizer], feed_dict=feed_dict_input)

        prediction = prediction > 0.5;
        training_accuracy.append(np.mean(prediction == y_batch))
    epoch_training_iou = [];
    for j in range(mini_batch_size):
        epoch_training_iou.append(iou.IoU_3D(y_batch[j, :, :, :], prediction[j, :, :, :]))
    training_iou.append(np.mean(epoch_training_iou))
    training_iou.append(np.mean(epoch_training_iou))
    print(np.mean(training_iou))

    print('epoch: '+str(epoch)+' loss: '+str(loss_epoch))
    prediction = np.squeeze(prediction)
    print(np.mean(prediction == y_batch))
    loss_history.append(loss_epoch);
    if(epoch%20 == 0 and epoch > 1): #save it occassionally in case we stop the run
        pickle.dump([loss_history, training_iou, training_accuracy], open('gray_scale_planes_training_data.p', 'wb'));
        saver.save(sess, './grayscale_plane_R2N2_model_weights', global_step=epoch)
        #we can artificially rename the checkpoint? probably not

        #predictions = tf.argmax(outputs, axis = 4)

save_path = saver.save(sess, "./"+class_name+".ckpt")
saver.save(sess, './grayscale_plane_R2N2_model_weights', global_step = epochs)


## run test batch
test_iou = list();
print('run test case')
test_samples, T,W,H,C = X_test.shape;
num_test_batches = int(test_samples/mini_batch_size);

for i in range(num_test_batches):
    start = i * mini_batch_size;
    end = (i + 1) * mini_batch_size;

    X_batch = X_test[start:end, :, :, :, :]

    X_batch = np.transpose(X_batch, axes=[1, 0, 2, 3, 4])

    ## convert this back to a list of arrs
    X_final = [X_batch[t, :, :, :, :] for t in range(T)];


    ## perform downsampling
    feed_dict_input = {i: d for i, d in zip(sequence, X_final)}; #size has to be fixed...
    feed_dict_input[label_placeholder] = y_test[start:end];
    test_loss= sess.run(loss, feed_dict=feed_dict_input)
    test_prediction = sess.run(outputs, feed_dict=feed_dict_input)
    test_prediction = test_prediction > 0.5;
    for k in range(mini_batch_size):
        print(iou.IoU_3D(y[k, :, :, :], test_prediction[k, :, :, :]))
        test_iou.append(iou.IoU_3D(y[k, :, :, :], test_prediction[k, :, :, :]))

print('test_loss')
print(test_loss)
print('number of test ex: '+str(len(test_iou)))
print(np.mean(test_iou))

pickle.dump([loss_history, training_iou, training_accuracy, test_iou], open('gray_scale_planes_training_data.p', 'wb'));




