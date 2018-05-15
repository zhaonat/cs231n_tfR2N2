'''
function designed to downsample the voxel model uniformly
'''

def downsample(voxel_model, stride):
    '''
    assumes that every thing is 3D
    :param voxel_model:
    :param stride:
    :return:
    '''
    new_model = voxel_model[0:-1:stride[0], 0:-1:stride[1], 0:-1:stride[2]]
    return new_model;
