import tensorflow as tf
import settings
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


export_dir = os.path.join(settings.ROOT_DIR, 'saved_models', 'grayscale_plane_2500');
# with tf.Session(graph=tf.Graph()) as sess:
#   tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)

from tensorflow.contrib import predictor

predict_fn = predictor.from_saved_model(export_dir)
print(predict_fn)


## import data
data_dir = os.path.join(settings.ROOT_DIR, 'data');
f = open(os.path.join(data_dir, 'planes_4045_batch', \
                      'planes_cropped_grayscale','planes_02691156_4045_batch_cropped_grayscale.p'), 'rb');
#f = open(os.path.join(data_dir, 'R2N2_128_batch_1.p'), 'rb');
class_name = 'planes';
X,y  = pickle.load(f);
## =========================================================
batch_size,T,H,W,C = X.shape;
mini_batch_size = 48;
NX = NY = NZ = 32


t_sample = 2;
sequence = []
for i in range(0,t_sample):
    sequence.append(tf.placeholder(tf.float32, shape = [mini_batch_size ,H,W,C], name = 'img_'+str(i)))

sequence = ['img_0:0', 'img_1:0'];
print(sequence)
start = 0;
end = mini_batch_size;
X_batch = X[start:end, :, :, :, :]
X_batch = np.transpose(X_batch, axes=[1, 0, 2, 3, 4])
y_batch = y[start:end, :, :, :]

## convert this back to a list of arrs

X_final = [X_batch[t, :, :, :, :] for t in range(t_sample)];

feed_dict_input = {i: d for i, d in zip(sequence, X_final)};
feed_dict_input['voxel'] = y_batch;
print(feed_dict_input.keys())

predictions = predict_fn(feed_dict_input)
print(predictions.keys())
sig_scores = predictions['outputs']
print(sig_scores.shape);
voxel = sig_scores > 0.5;

for i in range(len(y_batch)):
    fig = plt.figure();
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.voxels(voxel[i,:,:,:], edgecolor='black')
    plt.title('prediction')


    ax2 = fig.add_subplot(122, projection='3d')
    ax2.voxels(y_batch[i,:,:,:], facecolors = 'green', edgecolor='black')
    plt.title('original')
    plt.show()
