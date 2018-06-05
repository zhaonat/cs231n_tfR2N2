import os
import settings
import binvox_rw as brp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subsample_voxel as sv
import numpy as np

counter = 0;
sparsity_ratio = list();

shapecore = os.path.join(settings.ROOT_DIR,'data','ShapeNetCore.v2')
for collection_label in os.listdir(shapecore):
    if('.json' in collection_label):
        continue;
    #sample_rendering_set = os.path.join(settings.ROOT_DIR,'data','ShapeNetCore.v2', '02747177')
    sample_rendering_set = os.path.join(shapecore, collection_label);
    for model_id in os.listdir(sample_rendering_set):
        model_dir = os.path.join(sample_rendering_set, model_id, 'models');

        Big_3D_model_file  = os.path.join(model_dir, 'model_normalized.solid.binvox')
        with open(Big_3D_model_file, 'rb') as f:
            model = brp.read_as_3d_array(f)
        sub_image = model.data[:, 0:32, :]  # doing this crop is sufficient!
        sub_image = sv.downsample(sub_image, (4,1,4))
        sparsity_data = [np.sum(sub_image == 1)/np.prod(sub_image.shape), np.sum(model.data == 1)/np.prod(model.data.shape)]
        sparsity_ratio.append(sparsity_data)


        # fig = plt.figure()
        # ax = fig.add_subplot(1,2,1)
        # ax = fig.gca(projection='3d')
        # ax.voxels(model.data, edgecolor='black')
        # # fig = plt.figure();
        # # ax = fig.add_subplot(1,2,2)
        # # ax = fig.gca(projection='3d')
        # # ax.voxels(sub_image, edgecolor='black')
        # # plt.show()
        # plt.figure();
        # sampled = sv.downsample(sub_image, stride=[4, 1, 4])
        # ax = fig.gca(projection='3d')
        # ax.voxels(sampled, edgecolor='black')
        # plt.show()
        counter+=1;
        if(counter %10 == 0):
            print(counter)
        if(counter> 500):
            break;

sparsity_ratio = np.array(sparsity_ratio);
plt.figure();
plt.hist(sparsity_ratio);
plt.legend(['original', 'down-sampled'])
plt.xlabel('sparsity')
plt.ylabel('number of samples')
plt.title('sparsity distribution of voxel models')
plt.show()