import tensorflow as tf
import RecordLoader as rloader
from RecordUtil import converLabelsToInt


def test(tfrecordPath, batchSize, utilDict, modelGraph, model):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(modelGraph)
        saver.restore(sess, tf.train.latest_checkpoint(model))
        filename = tf.placeholder(tf.string, [None], name='filename')
        dataset = rloader.create_dataset(filename, 1, 200, batchSize)
        iterator = dataset.make_initializable_iterator()
        videoLens, labelLens, videos, labels = iterator.get_next()
        sess.run(iterator.initializer, feed_dict={filename: [tfrecordPath]})

        graph = tf.get_default_graph()
        videoInput = graph.get_tensor_by_name('video_input:0')
        videoLengths = graph.get_tensor_by_name('video_length:0')
        targets = graph.get_tensor_by_name('label_input:0')
        targetLengths = graph.get_tensor_by_name('label_Length:0')
        channel_keep_prob = graph.get_tensor_by_name('channel_keep_prob:0')
        cost = graph.get_tensor_by_name('out_loss:0')
        cer = graph.get_tensor_by_name('out_cer:0')
        _y = graph.get_tensor_by_name('out_decoded:0')
        while True:
            try:
                videoLen_batch, labelLen_batch, video_batch, label_batch = sess.run([videoLens, labelLens, videos, labels])
                label_batch = converLabelsToInt(utilDict, label_batch)
                loss, out_cer, _ = sess.run([cost, cer, _y], feed_dict={videoInput: video_batch,
                                                                        videoLengths: videoLen_batch,
                                                                        targets: label_batch,
                                                                        targetLengths: labelLen_batch,
                                                                        channel_keep_prob: 1.})
                print('Test: loss:{}, cer:{}'.format(loss, out_cer))
            except tf.errors.OutOfRangeError:
                break


BATCH_SIZE = 50
TEST_TFRECORD = 'put your tfrecord file here'
utilDict = rloader.loadUtilDict('utilDict.pkl')
modelGraph = 'put your model graph'
model = 'put your model here'
test(TEST_TFRECORD, BATCH_SIZE, utilDict, modelGraph, model)


