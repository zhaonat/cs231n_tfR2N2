import tensorflow as tf
import numpy as np
from modules import encoder
from modules import decoder
from modules import simple_lstm
from modules import my_3d_lstm
import settings
import os
import pickle
from data_processing import intersection_over_union as iou

from data_processing import pickle_to_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
'''
this takes a while to run... concatenates several batch files into one dataset...
'''
from data_processing import subsample_voxel
data_dir = os.path.join(settings.ROOT_DIR, 'data', 'gray_scale_shapenet');
y_total = None;

counter = 0;
max_counter = 400;
for file in os.listdir(data_dir):
    print('files processed: '+str(counter))
    if(counter> max_counter):
        break;
    f = open(os.path.join(data_dir, file), 'rb');

    batch_sample = pickle.load(f);
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

    ## concatenate into one large batch
    if(counter == 0):
        y_total = y

    else:
        y_total= np.concatenate((y_total, y), axis=0);
    print(y_total.shape)
    counter+=1;

y_data = np.reshape(y_total, (len(y_total), 32**3));

print(y_data.shape)
## do t-sne
X_embedded = TSNE(n_components=2).fit_transform(y_data)

plt.figure();
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.savefig('t-sne_voxels.png')
plt.show()