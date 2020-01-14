import tensorflow as tf
import numpy as np
import RecordLoader as rloader
from RecordUtil import converIntToLabels
from RecordConver import readVideo

def readVideos(filenames):
    videos = []
    videoLens = []
    for filename in filenames:
        video = readVideo(filename)
        videoLens.append(len(video))
        videos.append(video)

    videos = np.array(videos, dtype=np.uint8)
    return videos, videoLens


def predictVideos(videoPaths, utilDict, modelGraph, model):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(modelGraph)
        saver.restore(sess, tf.train.latest_checkpoint(model))
        graph = tf.get_default_graph()
        videoInput = graph.get_tensor_by_name('video_input:0')
        videoLengths = graph.get_tensor_by_name('video_length:0')
        channel_keep_prob = graph.get_tensor_by_name('channel_keep_prob:0')
        _y = graph.get_tensor_by_name('out_decoded:0')
        videos, videoLens = readVideos(videoPaths)
        y = sess.run([_y], feed_dict={videoInput: videos,
                                      videoLengths: videoLens,
                                      channel_keep_prob: 1.})
        result = converIntToLabels(utilDict, y[0])
    return result


utilDict = rloader.loadUtilDict('utilDict.pkl')
modelGraph = 'put your model graph'
model = 'put your model here'
print(predictVideos(['put your video path here'], utilDict, modelGraph, model))
