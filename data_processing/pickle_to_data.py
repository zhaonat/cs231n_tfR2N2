'''
unravels the mined pickle file into
X, Y
'''

import numpy as np
import pickle
import settings

def unravel_batch_pickle(batch_dictionary):
    '''
    :param batch_dictionary:
    :return: list of image sequences, list of the associated voxels
    '''
    X = list();
    y = list();
    for key in batch_dictionary:
        data = batch_dictionary[key];
        image_seq = data['image_sequence'];
        voxel_y = data['voxel_model'];
        X.append(image_seq);
        y.append(voxel_y);

    return X,y;