
'''
analyzes the shapenet core models and converts them to 3d numpy arrays
'''
import os
import gzip
import numpy as np
import binvox_rw as brp
import subsample_voxel as sv
import densify_voxel as dv
#from settings import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ROOT_DIR = 'D:\\Documents\\Classes\\CS231n\\cs231n_Project\\'
model_dir = os.path.join(ROOT_DIR, 'data', 'ShapeNetCore.v2')
rendering_dir = os.path.join(ROOT_DIR, 'data', 'ShapeNetRendering')
model_sample = os.path.join(model_dir, '02691156\\1a04e3eab45ca15dd86060f189eb133\\models');
import pickle;
file = os.path.join(model_sample, 'model_normalized.solid.binvox')
print(file)
with open(file, 'rb') as f:
    model = brp.read_as_3d_array(f)
    print(model.dims)
    image = model.data;
    # for i in range(20):
    #     plt.figure()
    #     plt.imshow(image[:,:,5*i]);
    #     plt.show()


## iterate through all subdirectories in 0269...

#data format
# we will store a dictionary with keys being all the model_ids from shapenet
#each dictionary entry will contain a dictionary with two entries
#first key is: image_sequence
# second key is: voxel_model

data_dict = dict();
counter = 1;
batch_counter = 1
data_dict = dict();
batch_size = 512

search_dirs = ['02691156', '02828884', '04530566', '04401088']
## cycle through the shapenet rendering...somehow we have many less renderings than cores...
for search_dir in os.listdir(os.path.join(rendering_dir)):
#
# for search_dir in search_dirs:

    for item in os.listdir(os.path.join(model_dir, search_dir)):
        counter += 1;

        #print('item: '+str(counter))
        model_id = item;
        if(model_id in data_dict.keys() ):
            continue;
        data_dict[model_id] = dict();


        #if the model id cannot be found
        if(not os.path.isdir(os.path.join(rendering_dir, search_dir, model_id))):
            continue;

        sequence_dir = os.path.join(rendering_dir, search_dir, model_id, 'rendering');
        #step 1: iterate through all images in the rendering directory
        image_sequence = list();
        for file in os.listdir(sequence_dir):
            if('.png' in file):
                image = mpimg.imread(os.path.join(sequence_dir, file));
                image_sequence.append(image);
        data_dict[model_id]['image_sequence'] = image_sequence;

        #step 2: load up the object model
        voxel_dir = os.path.join(model_dir, search_dir, model_id, 'models');
        file = os.path.join(voxel_dir, 'model_normalized.solid.binvox');
        with open(file, 'rb') as f:
            model = brp.read_as_3d_array(f)

        ## subsample the voxel model
        sub_image = model.data[:, 0:32, :] #doing this crop is sufficient!
        print(np.sum(sub_image) / np.prod(sub_image.shape))
        sampled = sv.downsample(sub_image, stride=[4, 1, 4])
        print('samples processessed: '+str(counter))
        data_dict[model_id]['voxel_model'] = sampled;
        if(counter % batch_size == 0):
            print('batch done, we have: '+str(counter)+' samples processed');
            print(data_dict.keys())
            pickle.dump(data_dict, open('R2N2_'+str(batch_size)+'_batch_'+str(batch_counter)+'origin_'+search_dir+'.p', 'wb'));
            batch_counter+=1;
            data_dict = dict();

        if(counter> 40*batch_size):
            break;

print('total_samples_processed: '+str(counter))
# print('batch done');
# print(data_dict.keys())
# pickle.dump(data_dict, open('R2N2_'+str(batch_size)+'_batch_'+str(batch_counter)+'.p', 'wb'));
# batch_counter+=1;
# data_dict = dict();