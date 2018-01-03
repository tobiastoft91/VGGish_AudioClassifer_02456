from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
import os, sys
tfrecords_filename = 'tfRecordsReal/val1.tfrecords'

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
embedding_tot =  np.zeros((1, 128))
embedding_labels_tot = []

for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    embedding_labels = int(example.features.feature['val1/labels']
                                 .int64_list
                                 .value[0])

    embedding_string = (example.features.feature['val1/embedding']
                                  .bytes_list
                                  .value[0])

    embedding_1d = np.fromstring(embedding_string, dtype=np.float32)
    #print(embedding_1d.shape)
    embedding_labels_tot.append(embedding_labels)
    #print(embedding_1d[0:10])
    reconstructed_embedding = embedding_1d.reshape((-1, 128))
    embedding_tot = np.append(embedding_tot,reconstructed_embedding, axis=0)

embedding_tot = np.delete(embedding_tot, 0, axis=0)
print(embedding_tot.shape)
print(len(embedding_labels_tot))
print(embedding_labels_tot[1:30])
#print(embedding_labels_tot)
