'''
function to visualize a 3D voxel model

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_vox(input):
    '''

    :param input: should be one Nx x Ny x Nz array
    :return:
    '''

    assert len(input.shape) == 3, 'wrong shape, check you have only one voxel model'

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(input, edgecolor='k')
