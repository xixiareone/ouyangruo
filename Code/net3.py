import os
import tensorflow as tf


model_path = '/home/ouyangruo/Documents/BiShe/Model/'

graph = tf.Graph()
def get_args():
    with graph.as_default():
        args = (
            tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
            tf.Variable(tf.zeros([32])),

            tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
            tf.Variable(tf.constant(1.0, shape=[64])),

            tf.Variable(tf.truncated_normal([3, 3, 64, 32], stddev=0.1)),
            tf.Variable(tf.constant(1.0, shape=[32])),

            tf.Variable(tf.truncated_normal([4 * 4 * 32, 64], stddev=0.1)),
            tf.Variable(tf.constant(1.0, shape=[64])),

            tf.Variable(tf.truncated_normal([64, 64], stddev=0.1)),
            tf.Variable(tf.constant(1.0, shape=[64])),

            tf.Variable(tf.truncated_normal([64, 10], stddev=0.1)),
            tf.Variable(tf.constant(1.0, shape=[10])),
        )
    return args

def model(data, args, is_training=True):
    conv = tf.nn.conv2d(data, args[0], [1, 1, 1, 1], padding='SAME')
    maxpool = tf.nn.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(maxpool + args[1])

    conv = tf.nn.conv2d(hidden, args[2], [1, 1, 1, 1], padding='SAME')
    maxpool = tf.nn.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(maxpool + args[3])

    conv = tf.nn.conv2d(hidden, args[4], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[5])

    conv = tf.nn.conv2d(hidden, args[6], [1, 1, 1, 1], padding='SAME')
    maxpool = tf.nn.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(maxpool + args[7])

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, args[8]) + args[9])

    hidden = tf.nn.relu(tf.matmul(hidden, args[10]) + args[11])

    return tf.matmul(hidden, args[12]) + args[13]