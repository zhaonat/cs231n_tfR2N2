import tensorflow as tf
import settings
import os
import pickle
import tensorflow as tf
import numpy as np

## import data
data_dir = os.path.join(settings.ROOT_DIR, 'data');
f = open(os.path.join(data_dir, 'planes_4045_batch', \
                      'planes_cropped_grayscale','planes_02691156_4045_batch_cropped_grayscale.p'), 'rb');
#f = open(os.path.join(data_dir, 'R2N2_128_batch_1.p'), 'rb');
class_name = 'planes';
X,y  = pickle.load(f);
## =========================================================
batch_size,T,H,W,C = X.shape;
mini_batch_size = 16;
NX = NY = NZ = 32

label_placeholder = tf.placeholder(tf.float32, shape = [mini_batch_size , NX,NY,NZ], name = 'voxel');
t_sample = 6;
sequence = []
for i in range(0,t_sample):
    sequence.append(tf.placeholder(tf.float32, shape = [mini_batch_size ,H,W,C], name = 'img_'+str(i)))

## ===============================================
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './input/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './input/train',
                           """Directory where to read model checkpoints.""")

saved_model_dir = os.path.join(settings.ROOT_DIR, 'saved_models')


sess=tf.Session()
writer = tf.summary.FileWriter('./', sess.graph)


#First let's load meta graph and restore weights
model_name = 'grayscale_plane_R2N2_model_weights-5.meta'
saver = tf.train.import_meta_graph(os.path.join(saved_model_dir, model_name))
saver.restore(sess,tf.train.latest_checkpoint(os.path.join(saved_model_dir)))
graph = tf.get_default_graph();

print([x for x in tf.get_default_graph().get_operations() if x.type == 'Placeholder'])
print(len([x for x in tf.get_default_graph().get_operations() if x.type == 'Placeholder']))
#check for the placeholders:


## list out all variables...
# graph = tf.get_default_graph()
# for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#     print(v)

## construct batch for feed_dict
X_batch = X[0:mini_batch_size, :, :, :, :]

X_batch = np.transpose(X_batch, axes=[1, 0, 2, 3, 4])
## convert this back to a list of arrs
X_final = [X_batch[t, :, :, :, :] for t in range(T)];

feed_dict_input = {i: d for i, d in zip(sequence, X_final)};
feed_dict_input[label_placeholder] = y[0:mini_batch_size]

## try to run the last layer
sess.run('classification_activation:0',feed_dict = feed_dict_input)

# ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print('Checkpoint found')
# else:
#     print('No checkpoint_500 found')

## run model on examples again
## visualize weights:
#tf.image_summary('conv1/filters', weights_transposed, max_images=3)
