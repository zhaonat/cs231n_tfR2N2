'''
analyzes the shapenet core models and converts them to 3d numpy arrays
'''
import os
import gzip
from scipy import ndimage
import numpy as np
import binvox_rw as brp
import subsample_voxel as sv
import densify_voxel as dv
# from settings import *
import PIL.Image
from skimage.measure import block_reduce
import pickle;

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = 'D:\\Documents\\Classes\\CS231n\\cs231n_Project\\'
model_dir = os.path.join(ROOT_DIR, 'data', 'ShapeNetCore.v2')
rendering_dir = os.path.join(ROOT_DIR, 'data', 'ShapeNetRendering')
# model_sample = os.path.join(model_dir, '02691156\\1a04e3eab45ca15dd86060f189eb133\\models');
#
# file = os.path.join(model_sample, 'model_normalized.solid.binvox')
# print(file)
# with open(file, 'rb') as f:
#     model = brp.read_as_3d_array(f)
#     print(model.dims)
#     image = model.data;
#     # for i in range(20):
#     #     plt.figure()
#     #     plt.imshow(image[:,:,5*i]);
#     #     plt.show()

## iterate through all subdirectories in 0269...

# data format
# we will store a dictionary with keys being all the model_ids from shapenet
# each dictionary entry will contain a dictionary with two entries
# first key is: image_sequence
# second key is: voxel_model

data_dict = dict();
counter = 1;
batch_counter = 1
data_dict = dict();
batch_size = 1024; #large number so we mine everything...

if (os.path.isfile('efficient_mined_keys.p')):
    already_mined_keys = pickle.load(open('efficient_mined_keys.p', 'rb'));
else:
    already_mined_keys = list();

## instead of seqeuntially going through directories, we will go through classes we want, get the sysnset id
# and then mine...



taxonomy_dict = {'02691156': 'planes', '04379243': 'table' , \
                 '03001627': 'chair', \
                 '02958343': 'car', '04256520': 'sofa'}

sequence_length = 6; #number of sequence images to extract

## cycle through the shapenet rendering...somehow we have many less renderings than cores...
for search_dir in taxonomy_dict.keys():
    print(search_dir+', '+taxonomy_dict[search_dir])
    # for search_dir in search_dirs:

    for item in os.listdir(os.path.join(model_dir, search_dir)):
        counter += 1;

        # print('item: '+str(counter))
        model_id = item;
        if (model_id in already_mined_keys):
            print('already_mined')
            continue;
        if (model_id in data_dict.keys()):
            continue;
        data_dict[model_id] = dict();
        already_mined_keys.append(model_id);

        # if the model id cannot be found
        if (not os.path.isdir(os.path.join(rendering_dir, search_dir, model_id))):
            continue;

        sequence_dir = os.path.join(rendering_dir, search_dir, model_id, 'rendering');
        # step 1: iterate through all images in the rendering directory
        image_sequence = list();
        sequence_counter = 0;
        for file in os.listdir(sequence_dir):
            if(sequence_counter> sequence_length):
                break;
            if ('.png' in file):
                # image = mpimg.imread(os.path.join(sequence_dir, file));
                rgba_image = PIL.Image.open(os.path.join(sequence_dir, file))
                rgb_image = rgba_image.convert('RGB')
                # print(np.asarray(rgb_image).shape)
                ## ================== CONVERT IMAGE TO GRAYSCALE ====================##
                image = np.asarray(rgb_image.convert('L'))

                # cannot do the bounding box because the picture sizes must be uniform at the END
                # cannot do the bounding box because the picture sizes must be uniform at the END
                # object_slices = ndimage.measurements.find_objects(image)
                # print(object_slices)
                # plt.imshow(image);
                # plt.show()
                # print(image.shape)
                ## ==================================================================##

                image_sequence.append(image);
                sequence_counter+=1;
        data_dict[model_id]['image_sequence'] = image_sequence;

        # step 2: load up the object model
        voxel_dir = os.path.join(model_dir, search_dir, model_id, 'models');
        file = os.path.join(voxel_dir, 'model_normalized.solid.binvox');
        if (not os.path.isfile(file)):
            continue;
        with open(file, 'rb') as f:
            model = brp.read_as_3d_array(f)

        ## don't apply any down-sampling
        sub_image = block_reduce(model.data, block_size=(4,4,4), func=np.max)
        print(np.sum(sub_image) / np.prod(sub_image.shape))
        sampled = sub_image;
        # fig = plt.figure();
        # ax = fig.gca(projection='3d')
        # ax.voxels(sampled, edgecolor='black')
        # plt.show()
        print('samples processessed: ' + str(counter))
        data_dict[model_id]['voxel_model'] = np.int8(sampled);
        if (counter % batch_size == 0):
            print('batch done, we have: ' + str(counter) + ' samples processed');
            print(data_dict.keys())
            pickle.dump(data_dict, open(
                'efficient_32_voxel_R2N2_' + str(batch_size) + '_batch_' + str(batch_counter) + 'class_' +\
                taxonomy_dict[search_dir] + '.p', 'wb'));
            batch_counter += 1;
            data_dict = dict();
            pickle.dump(already_mined_keys, open('efficient_mined_keys.p', 'wb'))

print('total_samples_processed: ' + str(counter))
# print('batch done');
# print(data_dict.keys())
# pickle.dump(data_dict, open('R2N2_'+str(batch_size)+'_batch_'+str(batch_counter)+'.p', 'wb'));
# batch_counter+=1;
# data_dict = dict();