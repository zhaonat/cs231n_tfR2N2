import tensorflow as tf
import settings
import os

saved_model_dir = os.path.join(settings.ROOT_DIR, 'saved_models')

import tensorflow as tf

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(os.path.join(saved_model_dir, 'R2N2_model_weights-400.meta'))
saver.restore(sess,tf.train.latest_checkpoint(os.path.join(saved_model_dir)))



## list out all variables...
graph = tf.get_default_graph()
for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(v)

## visualize weights:
#tf.image_summary('conv1/filters', weights_transposed, max_images=3)
