import os
import tensorflow as tf
import RecordUtil as rutil
import pickle
import cv2
import os
import numpy as np
from random import shuffle

def readVideo(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()
    return frames


def loadVideoNamesAndLabel(ListPath, cnt, flip=True):
    videoNames = list()
    labels = list()
    with open(ListPath, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            if i >= cnt:
                break
            line = line.replace('\n', '')
            videoName, label = line.split(',')
            videoNames.append(videoName)
            labels.append(label)
            if flip:
                videoNames.append('{}_flip'.format(videoName))
                labels.append(label)
            i += 1
        index_shuf = list(range(len(videoNames)))
        shuffle(index_shuf)
        shuffle_videoNames = list()
        shuffle_labels = list()
        for i in index_shuf:
            shuffle_videoNames.append(videoNames[i])
            shuffle_labels.append(labels[i])
        return shuffle_videoNames, shuffle_labels


def conver2Example(video, videoLen, label, labelLen):
    example = tf.train.Example(features=tf.train.Features(feature={
        'feature/videoLen': rutil.int64_feature(int(videoLen)),
        'feature/labelLen': rutil.int64_feature(int(labelLen)),
        'feature/video': rutil.bytes_feature(video.tostring()),
        'feature/label': rutil.bytes_feature(label.encode())
    }))
    return example


def conver2Record(VideoPath, ListPath, output, cnt):
    output_dir = os.path.dirname(output)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    with tf.python_io.TFRecordWriter(output) as tfWriter:
        i = 0;
        with open(ListPath, 'r', encoding='utf-8') as f:
            for line in f:
                if i >= cnt:
                    break
                line = line.replace('\n', '')
                videoName, label = line.split(',')
                frames = readVideo(os.path.join(VideoPath, videoName))
                frames = np.array(frames, dtype=np.uint8)
                example = conver2Example(frames, len(frames), label, len(label))
                tfWriter.write(example.SerializeToString())
                i += 1
                print('index:{}, shape:{}, label:{}, labelLen:{}'.format(i, frames.shape, label, len(label)))


def loadLabelGenerateWordDict(labelFile, cnt):
    with open(labelFile, 'r', encoding='utf-8') as f:
        chars = set(' ')
        maxLabelLen = 0
        line = f.readline()
        i = 0;

        while line:
            if i >= cnt:
                break
            line = line.replace('\n', '')
            file, label = line.split(',')
            if len(label) > maxLabelLen:
                maxLabelLen = len(label)
            for char in label:
                chars.add(char)
            line = f.readline()
            i += 1
        max = 0
        char2index = dict()
        for i, c in enumerate(chars):
            char2index[c] = i
            max = i
        char2index['blank'] = max + 1

        index2char = {i: c for c, i in char2index.items()}

        storeDict = dict()
        storeDict['char2index'] = char2index
        storeDict['index2char'] = index2char
        storeDict['maxLabelLen'] = maxLabelLen
        print(storeDict)
        with open('utilDict.pkl', 'wb') as p:
            pickle.dump(storeDict, p, pickle.HIGHEST_PROTOCOL)






