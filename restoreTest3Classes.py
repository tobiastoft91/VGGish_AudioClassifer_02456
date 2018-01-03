from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.path.join('.', '..'))
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

(embedding_test,embedding_labels_test) =  utils.tfRead('test')
print("tfRecord test uploaded!")

embedding_labels_test = utils.labelMinimizer(embedding_labels_test)
embedding_list = embedding_test

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

    y_pred = sess.run(op_to_restore, feed_dict)

    pred = sess.run(tf.argmax(y_pred, axis=1))
    #print("class predicion embedding 1:", pred)
    #print("real label: ",embedding_labels_test[0:100])

correct_pred = 0;
for i in range(0,len(pred)):
    if pred[i] == embedding_labels_test[i]:
        correct_pred+=1

acc = correct_pred/len(pred)
print("Test accuracy: ", acc)


# majority vote test:
from collections import Counter

correct_pred = 0;
idx = 0;
for n in range(1,round(len(pred)/10)):
    majorLabel= embedding_labels_test[idx:idx+9]
    majorPred = pred[idx:idx+9]
    cntLabel = Counter(majorLabel)
    cntPred = Counter(majorPred)

    idx = idx+10
    cLabel = cntLabel.most_common(1)[0]
    cPred = cntPred.most_common(1)[0]
    #print(cnt.most_common(1)[0])
    #print(cLabel[0])
    #print(cPred[0])
    if cLabel[0] == cPred[0]:
        correct_pred+=1

acc = correct_pred/round(len(pred)/10)
print("Test accuracy (major): ", acc)

conf_mat = confusion_matrix(embedding_labels_test,pred)
np.set_printoptions(precision=2)
conf_norm = conf_mat.astype('float')/conf_mat.sum(axis=1)[:,np.newaxis]
print(conf_norm*100)
print(sum(conf_mat))
print(sum(conf_mat)[1])

# 0 outdoor, 1 indoor, 2 vehicle
className = ["Outdoor","Indoor","Vehicle"]
# Plot normalized confusion matrix
plt.figure()
utils.plot_confusion_matrix(conf_mat, classes=className, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('myfig')
plt.show()
