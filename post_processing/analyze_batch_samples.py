from data_processing import visualize_voxel
import pickle
import os
import sys
import numpy as np
import settings
from data_processing import pickle_to_data
from data_processing import subsample_voxel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_dir = os.path.join(settings.ROOT_DIR, 'data', 'cars_7496_batch');
render_dir = os.path.join(settings.ROOT_DIR, 'data', 'ShapeNetRendering')
counter = 0;
max_counter = 100;
file = 'cars_02958343_7496_batch.p'
f = open(os.path.join(data_dir, file), 'rb');
X, y = pickle.load(f);
y0 = y;
X_nparr = np.array(X);
print(X_nparr.shape)
batch_size,T,H,W,C= X_nparr.shape;

#check out sample of images in each batch:
sequence_array = list();

sequence_length = 6;
num_samples = 10;
for i in range(num_samples):
    sequence = X_nparr[i, 0:sequence_length, :, :, :]; #T,W,H,C, want to reshape;
    print(sequence.shape)
    flattened = np.reshape(sequence, (W * sequence_length, H, C));
    flattened = np.transpose(flattened, axes = (1,0,2))
    sequence_array.append(flattened);
    print(sequence.shape)
    # plt.imshow(flattened);
    # plt.show();

sequence_array = np.array(sequence_array);
print(sequence_array.shape)
sequence_array = np.reshape(sequence_array, (num_samples*H, W * sequence_length, C))
print(sequence_array.shape)
sequence_array = np.transpose(sequence_array, axes = (1,0,2))
plt.figure(figsize = (20,20));
plt.imshow(sequence_array)
plt.savefig('sample_cars.png')
plt.show()
