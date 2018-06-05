import pickle
import matplotlib.pyplot as plt
import settings
import os

file = os.path.join(settings.ROOT_DIR, 'post_processing', 'gray_scale_planes_training_data.p')

[loss_history, training_iou, training_accuracy, test_iou] = pickle.load(open(file, 'rb'));

plt.figure();
plt.plot(loss_history);

plt.figure();
plt.plot(training_iou);
plt.show()
