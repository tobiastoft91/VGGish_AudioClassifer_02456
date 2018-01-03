
from __future__ import print_function

import numpy as np
#from scipy.io import wavfile
#import six
import tensorflow as tf
import os, sys

import utils
import vggish_params
import vggish_slim


def main(_):
  (train_addrs,train_labels, val_addrs,val_labels, test_addrs,test_labels) = utils.adressLabelSort('sortedTestAudio2')
  addr = train_addrs
  embedding_labels = train_labels
  print('number of addr: ',len(addr))
  print('number of labels: ',len(embedding_labels))

  (examples_batch,embedding_labels) = utils._get_batch(addr,embedding_labels)

  tfrecords_filename = 'Evalval1.tfrecords'
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)


  # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
  config = tf.ConfigProto()
  #config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allocator_type = 'BFC'
  #config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.90

  with tf.Graph().as_default(), tf.Session(config=config) as sess:

    vggish_slim.define_vggish_slim(training=False) # Defines the VGGish TensorFlow model.
    vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt') # Loads a pre-trained VGGish-compatible checkpoint.

    # locate input and output tensors.
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    feed_dict = {features_tensor: examples_batch}

    [embedding_batch] = sess.run([embedding_tensor], feed_dict=feed_dict)

    print('example_batch shape: ', examples_batch.shape)
    print('embedding_batch shape: ', embedding_batch.shape)
    print('labels_batch shape: ', len(embedding_labels))


    # store the data to the TFRecords file.
    for i in range(len(embedding_batch)):
        embedding = embedding_batch[i]


        # convert into proper data type:
        embedding_length = embedding_labels[i] # embedding.shape[0]
        embedding_raw = embedding.tostring()

        # Create a feature
        feature = {'Evalval1/labels': utils._int64_feature(embedding_length),
                   'Evalval1/embedding':  utils._bytes_feature(embedding_raw)}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
  tf.app.run()
