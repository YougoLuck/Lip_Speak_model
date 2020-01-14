import tensorflow as tf
import RecordLoader as rloader
from RecordUtil import converLabelsToInt
from Model4 import Model4
from Model3 import Model3
from Model2 import Model2
from Model1 import Model1


def train(tfrecordPath, epoch, batchSize, repeat, utilDict, m, ckptPath):
    with tf.Session() as sess:
        filename = tf.placeholder(tf.string, [None], name='filename')
        dataset = rloader.create_dataset(filename, repeat, 300, batchSize)
        iterator = dataset.make_initializable_iterator()
        videoLens, labelLens, videos, labels = iterator.get_next()
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch):
            sess.run(iterator.initializer, feed_dict={filename: [tfrecordPath]})
            step = 0
            while True:
                try:
                    videoLen_batch, labelLen_batch, video_batch, label_batch = sess.run([videoLens, labelLens, videos, labels])
                    label_batch = converLabelsToInt(utilDict, label_batch)
                    feedDict = m.get_feed_dict(videoLen_batch, labelLen_batch, video_batch, label_batch, isTrain=True)
                    loss, _, cer = sess.run([m.cost, m.train_op, m.cer],  feed_dict=feedDict)
                    print('Train: epoch:{}, step:{}, loss:{}, cer:{}'.format(epoch, step, loss, cer))
                    step += 1
                except tf.errors.OutOfRangeError:
                    if epoch and epoch % 5 == 0:
                        saver.save(sess, ckptPath, write_meta_graph=True, global_step=epoch)
                    break


MAX_EPOCH = 40000
REPEAT = 1
BATCH_SIZE = 50
TRAIN_TFRECORD = 'put your tfrecord file here'
utilDict = rloader.loadUtilDict('utilDict.pkl')
m = Model1(utilDict)
train(TRAIN_TFRECORD, MAX_EPOCH, BATCH_SIZE, REPEAT, utilDict, m, '')
