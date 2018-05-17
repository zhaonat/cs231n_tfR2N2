import tensorflow as tf
import numpy as np
from modules import encoder
from modules import decoder
from modules import simple_lstm
import settings
import os
import pickle
from data_processing import pickle_to_data
from data_processing import subsample_voxel as sv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_processing import subsample_voxel
data_dir = os.path.join(settings.ROOT_DIR, 'data_sample');

for i in range(8,10):
    f = open(os.path.join(data_dir, 'R2N2_9_batch_'+str(i)+'.p'), 'rb');

    batch_sample = pickle.load(f);
    print(batch_sample.keys())
    X, y = pickle_to_data.unravel_batch_pickle(batch_sample)

    X_nparr = np.array(X);
    X_seq = X[0];
    fig = plt.figure(figsize = (30,5));
    counter = 1;
    for image in X_seq:

        index = '1'+str(5)+str(counter);
        print(index)
        plt.subplot(int(index))
        plt.imshow(X_seq[counter*4])
        plt.title('2D view '+str(counter))
        counter +=1;
        if(counter > 4):
            break;
    ax = fig.add_subplot(1,5,5, projection='3d')
    ax.voxels(y[0], edgecolor='k')
    ax.set(title = '3D Model')
    plt.show()
    # for m in range(len(y)):
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     sub_image = y[m]
    #     print(np.sum(sub_image)/np.prod(sub_image.shape))
    #     ax.voxels(sub_image, edgecolor='k')
    #     plt.show()