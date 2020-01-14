import tensorflow as tf
import numpy as np


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def converLabelsToInt(utilDict, labels):
    char2index = utilDict['char2index']
    maxLabelLen = utilDict['maxLabelLen']
    intLabels = []
    for label in labels:
        label = str(label, encoding='utf-8')
        intLabel = []
        for char in label:
            intLabel.append(char2index[char])
        intLabels.append(list(intLabel + [-1] * (maxLabelLen - len(intLabel))))
    return np.array(intLabels)


def converIntToLabels(utilDict, labels):
    index2char = utilDict['index2char']
    stringLabels = []
    for label in labels:
        string = ''
        for intLabel in label:
            if intLabel == -1:
                continue
            string = string + index2char[intLabel]
        stringLabels.append(string)
    return stringLabels
