from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.path.join('.', '..'))
import utils
import tensorflow as tf
import numpy as np

#(embedding_train,embedding_labels_train) = utils.read_tfrecords_train('tfRecords10procent/train.tfrecords')
#(embedding_val,embedding_labels_val) = utils.read_tfrecords_val('tfRecords10procent/val.tfrecords')

(e1_train,l1_train) =  utils.tfRead('train1')
print("tfRecord train1 uploaded!")
(e2_train,l2_train) =  utils.tfRead('train2')
print("tfRecord train2 uploaded!")
(e3_train,l3_train) =  utils.tfRead('train3')
print("tfRecord train3 uploaded!")
(e4_train,l4_train) =  utils.tfRead('train4')
print("tfRecord train4 uploaded!")
(e5_train,l5_train) =  utils.tfRead('train5')
print("tfRecord train5 uploaded!")
(e6_train,l6_train) =  utils.tfRead('train6')
print("tfRecord train6 uploaded!")
embedding_train= np.concatenate((e1_train, e2_train, e3_train, e4_train, e5_train, e6_train), axis=0)
print("Train embedding shape: ",embedding_train.shape)

embedding_labels_train = np.concatenate((l1_train, l2_train, l3_train, l4_train, l5_train, l6_train), axis=0)
print(embedding_labels_train.shape)

(e1_val,l1_val) =  utils.tfRead('val1')
print("tfRecord val1 uploaded!")
(e2_val,l2_val) =  utils.tfRead('val2')
print("tfRecord val2 uploaded!")

embedding_val= np.concatenate((e1_val, e2_val), axis=0)
print("Val embedding shape: ",embedding_val.shape)

embedding_labels_val = np.concatenate((l1_val,l2_val), axis=0)
print(embedding_labels_val.shape)

#One hot encoding
embedding_labels_train = utils.OnehotEnc(embedding_labels_train)
embedding_labels_val = utils.OnehotEnc(embedding_labels_val)

#print(embedding_labels_train[1])
#print(embedding_labels_train.shape)

num_features = embedding_train[0].shape[0]
num_classes = embedding_labels_train[0].shape[0]
print(num_features)
print(num_classes)


## Define placeholders
x_pl = tf.placeholder(tf.float32, shape=[None, num_features], name='xPlaceholder')
y_pl = tf.placeholder(tf.float32, shape=[None, num_classes], name='yPlaceholder')


## Define the model
def build_model(data_in):

    input_dim = num_features
    output_dim = num_classes
    hl1_dim = 100 # Number of neurons in hidden layer 1.

    # Initializer that generates a truncated normal distribution
    weight_initializer = tf.truncated_normal_initializer(stddev=0.1)

    w_hl1 = tf.get_variable('w_hl1', [input_dim, hl1_dim], initializer=weight_initializer)
    b_hl1 = tf.get_variable('b_hl1', [hl1_dim], initializer=tf.constant_initializer(0.0))

    hl_1 = tf.matmul(data_in, w_hl1) + b_hl1
    hl_1 = tf.nn.relu(hl_1) # activation function


    w_output = tf.get_variable('w_output', [hl1_dim, output_dim], initializer=weight_initializer)
    b_output = tf.get_variable('b_output', [output_dim], initializer=tf.constant_initializer(0.0))


    l_2 = tf.matmul(hl_1, w_output) + b_output
    prediction = tf.nn.softmax(l_2,name="op_to_restore")

    return prediction



### Implement training ops

prediction = build_model(x_pl)

# 1) Define cross entropy loss
cross_entropy = -tf.reduce_sum(y_pl * tf.log(prediction), reduction_indices=[1])
cross_entropy = tf.reduce_mean(cross_entropy)

# 2) Define the training op
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 3) Define accuracy op
    # This is used to monitor the performance during training, but doesn't have any influence on training.
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y_pl, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




num_epochs =2000
batch_size = 125

# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

train_cost, val_cost, train_acc, val_acc = [],[],[],[]
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    try:
        sess.run(tf.global_variables_initializer())
        for e in range(num_epochs):
            (x_tr,y_tr) = utils.next_batch(batch_size,embedding_train,embedding_labels_train)
            #x_tr, y_tr = mnist_data.train.next_batch(batch_size)
            (x_val,y_val) = utils.next_batch(batch_size,embedding_val,embedding_labels_val)
            #x_val, y_val = mnist_data.validation.next_batch(batch_size)

            # Traning optimizer
            feed_dict_train = {x_pl: x_tr, y_pl: y_tr}

            # running the train_op
            res = sess.run( [train_op, cross_entropy, accuracy], feed_dict=feed_dict_train)

            train_cost += [res[1]]
            train_acc += [res[2]]

            # Validation:
            feed_dict_valid = {x_pl: x_val, y_pl: y_val}

            res = sess.run([cross_entropy, accuracy], feed_dict=feed_dict_valid)
            val_cost += [res[0]]
            val_acc += [res[1]]

            #3) Print training summaries
            if e % 100 == 0:
                print("Epoch %i, Train Cost: %0.3f\tVal Cost: %0.3f\t Val acc: %0.3f" \
                      %(e, train_cost[-1],val_cost[-1],val_acc[-1]))


        # Save the output of the network to a local place
        saver = tf.train.Saver()
        saver.save(sess, "C:/tmp/audio_classifier10procent")

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

print('Done')
