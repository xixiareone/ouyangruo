import os
import tensorflow as tf


model_path = '/home/ouayangruo/Documents/BiShe/model/'

graph = tf.Graph()
def get_args():
    with graph.as_default():
        args = (
            tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
            tf.Variable(tf.zeros([32])),

            tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([3, 3, 64, 32], stddev=0.1)),
            tf.Variable(tf.zeros([32])),

            tf.Variable(tf.truncated_normal([4 * 4 * 32, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([64, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([64, 10], stddev=0.1)),
            tf.Variable(tf.zeros([10])),
        )
    return args

def model(data):
    conv = tf.nn.conv2d(data, layer0_weights, [1, 1, 1, 1], padding='SAME')
    maxpool = tf.nn.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(maxpool + layer0_biases)

    conv = tf.nn.conv2d(hidden, layer1_weights, [1, 1, 1, 1], padding='SAME')
    maxpool = tf.nn.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(maxpool + layer1_biases)

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)

    conv = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding='SAME')
    maxpool = tf.nn.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(maxpool + layer3_biases)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)

    hidden = tf.nn.relu(tf.matmul(hidden, layer5_weights) + layer5_biases)

    return tf.matmul(hidden, layer6_weights) + layer6_biases
