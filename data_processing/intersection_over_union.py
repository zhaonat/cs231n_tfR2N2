import numpy as np

def IoU_3D(ground_truth, prediction):
    indices_gt = np.asarray(np.where(ground_truth == 1)).T
    indices_p = np.asarray(np.where(prediction == 1)).T
    Nx, Ny, Nz = ground_truth.shape;
    asdf = dict();
    intersection = 0;
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if(ground_truth[i,j,k] == prediction[i,j,k] == 1):
                    intersection+=1;
                if((i,j,k) not in asdf.keys() and (ground_truth[i,j,k] == 1 or prediction[i,j,k] == 1)):
                    asdf[(i,j,k)] = 1;

    return intersection/len(asdf);

## test

# ground_truth = np.zeros((20, 20, 20));
# prediction = np.zeros((20, 20, 20));
# ground_truth[5:15, 5:15, 5:15] = 1;
# prediction[4:14, 4:14, 4:14] = 1;
#
# print(IoU_3D(ground_truth, prediction))


