from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.path.join('.', '..'))
import utils
import tensorflow as tf
import numpy as np

(e1_test,l1_test) =  utils.tfRead('test1')
print("tfRecord test1 uploaded!")
(e2_test,l2_test) =  utils.tfRead('test2')
print("tfRecord test2 uploaded!")

embedding_test= np.concatenate((e1_test,e2_test), axis=0)
print("Train embedding shape: ",embedding_test.shape)

embedding_labels_test = np.concatenate((l1_test,l2_test), axis=0)
print(embedding_labels_test.shape)

print(embedding_labels_test[195])
embedding_labels_test = utils.labelMinimizer(embedding_labels_test)
embedding_labels_test = utils.OnehotEnc(embedding_labels_test)


#print(embedding_test[1])
print(embedding_labels_test[195])


embedding_list = [embedding_test[195]]


gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
 # load the trained network from a local drive
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
#First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph("C:/tmp/audio_classifier.meta")
    saver.restore(sess,tf.train.latest_checkpoint('C:/tmp/'))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    x_pl = graph.get_tensor_by_name("xPlaceholder:0")
    feed_dict = {x_pl: embedding_list}

    #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    y_pred = sess.run(op_to_restore, feed_dict)[0]

    print(y_pred)
    print("class predicion embedding 1:",sess.run(tf.argmax(y_pred, axis=0)))
