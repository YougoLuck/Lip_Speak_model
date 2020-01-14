import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class Model1:

    channel_dropout = 0.5
    hidden_size = 256
    layers = 2
    beam_width = 60
    maxVidLen = 75

    def __init__(self, utilDict):
        self.maxLabelLen = utilDict['maxLabelLen']
        self.char2index = utilDict['char2index']
        self.initialize()
        self.build_graph()

    def initialize(self):
        # Inputs
        self.videoInput = tf.placeholder(shape=[None, self.maxVidLen, 66, 100, 3], dtype=tf.float32, name='video_input')

        self.videoLengths = tf.placeholder(shape=[None], dtype=tf.int32, name='video_length')

        self.targets = tf.placeholder(shape=[None, self.maxLabelLen], dtype=tf.int32, name='label_input')

        self.targetLengths = tf.placeholder(shape=[None], dtype=tf.int32, name='label_Length')

        self.channel_keep_prob = tf.placeholder(dtype=tf.float32, name='channel_keep_prob')


    def conv3d_layer(self, inputs, filters, kernel, strides, pool_size):

        conv = tf.layers.conv3d(inputs      = inputs,
                                filters     = filters,
                                kernel_size = kernel,
                                strides     = strides,
                                padding     = 'same',
                                activation  =  tf.nn.relu)

        # batch = tf.layers.batch_normalization(conv, training=self.isTrain, name='{}_normal'.format(name))


        pool = tf.layers.max_pooling3d(inputs      = conv,
                                       pool_size   = pool_size,
                                       strides     = pool_size)

        drop = tf.nn.dropout(x           = pool,
                             keep_prob   = self.channel_keep_prob)

        return drop

    def multi_cell(self):
        return rnn.MultiRNNCell([self.single_cell() for _ in range(self.layers)])

    def single_cell(self):
        cell = rnn.GRUCell(self.hidden_size)
        return cell

    def get_feed_dict(self, vidLens, labelLens, videos, labels, isTrain=True):
        feedDict = dict()
        if isTrain:
            feedDict[self.channel_keep_prob] = 1. - self.channel_dropout
        else:
            feedDict[self.channel_keep_prob] = 1.
        feedDict[self.videoLengths] = vidLens
        feedDict[self.targetLengths] = labelLens
        feedDict[self.targets] = labels
        feedDict[self.videoInput] = videos
        return feedDict

    def to_sparse(self, tensor, sparse_val=-1):
        sparse_inds = tf.where(tf.not_equal(tensor, sparse_val))
        sparse_vals = tf.gather_nd(tensor, sparse_inds)
        dense_shape = tf.shape(tensor, out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


    def build_graph(self):
        inputs = tf.to_float(self.videoInput)
        conv1 = self.conv3d_layer(
                inputs    = inputs,
                filters   = 32,
                kernel    = (3,5,5),
                strides   = (1,2,2),
                pool_size = (1,2,2))

        conv2 = self.conv3d_layer(
                inputs    = conv1,
                filters   = 64,
                kernel    = (3,5,5),
                strides   = (1,1,1),
                pool_size = (1,2,2))

        conv3 = self.conv3d_layer(
                inputs    = conv2,
                filters   = 96,
                kernel    = (3,3,3),
                strides   = (1,1,1),
                pool_size = (1,2,2))

        # Prepare for RNN
        cnn_out = tf.reshape(conv3, shape=(-1, self.maxVidLen, 6 * 4 * 96))
        cnn_out = tf.transpose(cnn_out, perm=[1, 0, 2])

        # LSTM layer
        lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw         = self.multi_cell(),
                cell_bw         = self.multi_cell(),
                inputs          = cnn_out,
                sequence_length = self.videoLengths,
                time_major      = True,
                dtype           = tf.float32)

        lstm_out = tf.concat(lstm_out, 2)

        # Output layer
        logits = tf.layers.dense(lstm_out, len(self.char2index))

        # Create train op
        sparse_targets = self.to_sparse(self.targets, -1)

        self.cost = tf.reduce_mean(tf.nn.ctc_loss(labels=sparse_targets,
                                                  inputs=logits,
                                                  sequence_length=self.videoLengths,
                                                  time_major=True,
                                                  ignore_longer_outputs_than_inputs=True), name='out_loss')

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

        # Create error rate op
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, self.videoLengths, beam_width = self.beam_width)
        self.cer = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_targets), name='out_cer')
        self.a = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values, default_value=-1, name='out_decoded')
