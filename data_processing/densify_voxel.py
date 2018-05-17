import numpy as np
'''
attempt to find minimum bounding box of a voxel
'''
import os
import pickle


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    new_image = img[rmin:rmax, cmin:cmax, zmin:zmax]
    return new_image

#test
#from data_processing import pickle_to_data
# data_dir = os.path.join(settings.ROOT_DIR, 'data_sample');
# f = open(os.path.join(data_dir, 'R2N2_9_batch_3.p'), 'rb');
# batch_sample = pickle.load(f);
# X, y = pickle_to_data.unravel_batch_pickle(batch_sample)
# image = y[0].astype(int)
#
# print(bbox2_3D(image));
# rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(image)
# sub_image = image[rmin:rmax, cmin:cmax, zmin:zmax]
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# sub_image = image[rmin:rmax, cmin:cmax, zmin:zmax ]
# fig = plt.figure(figsize = (20,8))
# ax = fig.add_subplot(1,2, 1, projection='3d')
# ax.voxels(sub_image, edgecolor='k')
# plt.show()