import tensorflow as tf
import pickle

# ====================================Data Property====================================
VIDEO_WIDTH = 100
VIDEO_HEIGHT = 66
VIDEO_CHANNEL = 3
# =====================================================================================

def _parse_function(example_proto):
    features = {'feature/videoLen': tf.FixedLenFeature((), tf.int64),
                'feature/labelLen': tf.FixedLenFeature((), tf.int64),
                'feature/video': tf.FixedLenFeature((), tf.string),
                'feature/label': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    videoLen = tf.cast(parsed_features['feature/videoLen'], tf.int32)
    labelLen = tf.cast(parsed_features['feature/labelLen'], tf.int32)
    label = parsed_features['feature/label']
    video = tf.decode_raw(parsed_features['feature/video'], tf.uint8)
    video = tf.reshape(video, (videoLen, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNEL))
    return videoLen, labelLen, video, label


def create_dataset(filepath, repeat, shuffle, batch_size):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size)
    return dataset

def loadUtilDict(filePath):
    with open(filePath, 'rb') as f:
        storeDict = pickle.load(f)
        return storeDict
