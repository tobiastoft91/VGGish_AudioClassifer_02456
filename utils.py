import tensorflow as tf
import os, sys
import numpy as np
from random import shuffle

import glob
import vggish_input
import vggish_params

def labelMinimizer(data):
    # 0 outdoor, 1 indoor, 2 vehicle
    for i in range(0,len(data)):

        if (data[i] == 0):
            data[i] = 0
        elif (data[i] == 1):
            data[i] = 2
        elif (data[i] == 2):
            data[i] = 1
        elif (data[i] == 3):
            data[i] = 2
        elif (data[i] == 4):
            data[i] = 1
        elif (data[i] == 5):
            data[i] = 2
        elif (data[i] == 6):
            data[i] = 1
        elif (data[i] == 7):
            data[i] = 1
        elif (data[i] == 8):
            data[i] = 1
        elif (data[i] == 9):
            data[i] = 1 # metro station
        elif (data[i] == 10):
            data[i] = 1
        elif (data[i] == 11):
            data[i] = 0
        elif (data[i] == 12):
            data[i] = 0
        elif (data[i] == 13):
            data[i] = 2
        elif (data[i] == 14):
            data[i] = 2

    return data

def OnehotEnc(data):
    from numpy import array
    from numpy import argmax
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    # define example
    #data = [0,0,0,0,1,1,1,1,2,2,2,2,4,4,4,4]
    values = array(data)
    #print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    # invert first example
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[1, :])])
    #print(inverted)
    return(onehot_encoded)


def tfRead(fileName):
    tfrecords_filename = 'tfRecordsReal/'+fileName+'.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    embedding_tot =  np.zeros((1, 128))
    embedding_labels_tot = []

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        embedding_labels = int(example.features.feature[fileName+'/labels']
                                     .int64_list
                                     .value[0])

        embedding_string = (example.features.feature[fileName+'/embedding']
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

    return(embedding_tot,embedding_labels_tot)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _get_batch(addr,embedding_labels):
  feat_tot = np.zeros((1, 96, 64))
  labels_tot = []

  for path in addr:
    feat = vggish_input.wavfile_to_examples(path)
    feat_tot = np.concatenate((feat_tot,feat))

  feat_tot = np.delete(feat_tot, 0, axis=0)

# not nice way of extending the label vector.... :
  for label in embedding_labels:
      lab = [i * label for i in [1,1,1,1,1,1,1,1,1,1]]
      labels_tot.extend(lab)

  print("Audio files uploaded!")

  return(feat_tot,labels_tot)

def next_batch(batch_size,idx_epochs, epochs_completed, data, labels):
    # returns num random features and labels
#    print("data before: ", data[1:5,1])
    if (idx_epochs == 0):
        perm0 = np.arange(0 , len(data))
        np.random.shuffle(perm0)
        data = data[perm0]
        labels = labels[perm0]
        #print("data after: ", data[1:5,1])

    start = idx_epochs*batch_size

    if (start+batch_size > len(data)):
        #print("batch longer!")
        # shuffle
        perm = np.arange(0 , len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        start=0
        end = start+batch_size-1

        idx_epochs = 1
        epochs_completed += 1
    else:
        end = start+batch_size-1
        idx_epochs += 1
        #print("start: ",start," end: ",end )
    return np.asarray(data[start:end]), np.asarray(labels[start:end]), epochs_completed, idx_epochs, data, labels

def next_batch_shuffle(num, data, labels):
    # returns num random features and labels
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def adressLabelSort(data_root):

    shuffle_data = True  # shuffle the addresses before saving
    train_addrs = []
    train_labels = []
    val_addrs = []
    val_labels = []
    test_addrs = []
    test_labels = []

    labels = []
    addrs_tot = []
    data_labels = os.listdir(data_root)

    for Label in data_labels:
        data_path = data_root + '/' + Label + '/*.wav'

        # read addresses and labels from the 'train' folder
        addrs = glob.glob(data_path)
        for addr in addrs:
            addrs_tot.append(addr)

            if 'beach' in addr:
                labels.append(0)
            elif 'bus' in addr:
                labels.append(1)
            elif 'caferestaurant' in addr:
                labels.append(2)
            elif 'car' in addr:
                labels.append(3)
            elif 'city_center' in addr:
                labels.append(4)
            elif 'forest_path' in addr:
                labels.append(5)
            elif 'grocery_store' in addr:
                labels.append(6)
            elif 'home' in addr:
                labels.append(7)
            elif 'library' in addr:
                labels.append(8)
            elif 'metro_station' in addr:
                labels.append(9)
            elif 'office' in addr:
                labels.append(10)
            elif 'park' in addr:
                labels.append(11)
            elif 'residential_area' in addr:
                labels.append(12)
            elif 'train' in addr:
                labels.append(13)
            elif 'tram' in addr:
                labels.append(14)


        # to shuffle data
    if shuffle_data:
        c = list(zip(addrs_tot, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the hata into 60% train, 20% validation, and 20% test
    train_addrs = addrs[0:int(1*len(addrs))]
    train_labels = labels[0:int(1*len(labels))]
    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = labels[int(0.8*len(labels)):]

    return train_addrs,train_labels, val_addrs,val_labels, test_addrs,test_labels




print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
