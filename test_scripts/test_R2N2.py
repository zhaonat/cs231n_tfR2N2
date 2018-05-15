import tensorflow as tf
import numpy as np
from modules import encoder
from modules import decoder
from modules import simple_lstm
import settings
import os
import pickle
from data_processing import pickle_to_data
import matplotlib.pyplot as plt

data_dir = os.path.join(settings.ROOT_DIR, 'data_sample');
f = open(os.path.join(data_dir, 'R2N2_9_batch_data.p'), 'rb');

batch_sample = pickle.load(f);
print(batch_sample.keys())
X, y = pickle_to_data.unravel_batch_pickle(batch_sample)
print(len(X))

sample_image = X[0][0]; print(sample_image.shape)
sample_model = y[0]; print(sample_model.shape)
plt.figure();
plt.imshow(sample_image[:,:,0:3]);
plt.show()

## specify necessary parameters
batch_size = 4;
[H,W,C] = sample_image.shape;
[X,Y,Z] = sample_model.shape;

image_placeholder= tf.placeholder(tf.float32, shape = [batch_size, H,W,C])
label_placeholder = tf.placeholder(tf.float32, shape = [batch_size, X,Y,Z]);

## specify total graph
sequence = [image_placeholder for i in range(5)]
encoded_sequence = list();
for image in sequence:
    encoded_out = encoder.encoder(image);
    encoded_sequence.append(encoded_out);

## after we use the encoder, we should have a sequence of dense outputs
conv_lstm_hidden = simple_lstm.simple_lstm(encoded_sequence);

##decode the lstm output, which is just the hidden state, 3D
# pass it through the decoder
logits = decoder.decoder(conv_lstm_hidden)

## define loss function
outputs = tf.contrib.layers.softmax(logits)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=label_placeholder,
    logits=logits,
)
loss = tf.layers.flatten(loss)
loss = tf.reduce_mean(loss);

predictions = tf.argmax(outputs, axis = 4)