import os
import tensorflow as tf


model_path = '/model/'

graph = tf.Graph()
def get_args():
    with graph.as_default():
        args = (
            tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.1)),
            tf.Variable(tf.zeros([16])),

            tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1)),
            tf.Variable(tf.zeros([32])),

            tf.Variable(tf.truncated_normal([1, 1, 32, 16], stddev=0.1)),
            tf.Variable(tf.zeros([16])),

            tf.Variable(tf.truncated_normal([3, 3, 16, 128], stddev=0.1)),
            tf.Variable(tf.zeros([128])),

            tf.Variable(tf.truncated_normal([1, 1, 128, 16], stddev=0.1)),
            tf.Variable(tf.zeros([16])),

            tf.Variable(tf.truncated_normal([3, 3, 16, 128], stddev=0.1)),
            tf.Variable(tf.zeros([128])),

            tf.Variable(tf.truncated_normal([1, 1, 128, 32], stddev=0.1)),
            tf.Variable(tf.zeros([32])),

            tf.Variable(tf.truncated_normal([3, 3, 32, 256], stddev=0.1)),
            tf.Variable(tf.zeros([256])),

            tf.Variable(tf.truncated_normal([1, 1, 256, 32], stddev=0.1)),
            tf.Variable(tf.zeros([32])),

            tf.Variable(tf.truncated_normal([1, 1, 32, 256], stddev=0.1)),
            tf.Variable(tf.zeros([256])),

            tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([3, 3, 64, 512], stddev=0.1)),
            tf.Variable(tf.zeros([512])),

            tf.Variable(tf.truncated_normal([1, 1, 512, 64], stddev=0.1)),
            tf.Variable(tf.zeros([64])),

            tf.Variable(tf.truncated_normal([3, 3, 64, 512], stddev=0.1)),
            tf.Variable(tf.zeros([512])),

            tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1)),
            tf.Variable(tf.zeros([128])),

            tf.Variable(tf.truncated_normal([1, 1, 128, 1000], stddev=0.1)),
            tf.Variable(tf.zeros([1000])),

            tf.Variable(tf.truncated_normal([14*14*1000, 1000], stddev=0.1)),
            tf.Variable(tf.zeros([1000])),

            tf.Variable(tf.truncated_normal([1000, 17], stddev=0.1)),
            tf.Variable(tf.zeros([17])),
              
            
        )
    return args

def model(data, args, is_training=True):
    conv = tf.nn.conv2d(data, args[0], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[1])
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1],[1, 2, 2, 1], padding='SAME')
    print ("1:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[2], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[3])
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    print ("2:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[4], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[5])
    print ("3:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[6], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[7])
    print ("4:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[8], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[9])
    print ("5:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[10], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[11])
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1],[1, 2, 2, 1], padding='SAME')
    print ("6:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[12], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[13])
    print ("7:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[14], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[15])
    print ("8:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[16], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[17])
    print ("9:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[18], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[19])
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1],[1, 2, 2, 1], padding='SAME')
    print ("10:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[20], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[21])
    print ("11:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[22], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[23])
    print ("12:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[24], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[25])
    print ("13:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[26], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[27])
    print ("14:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[28], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[29])
    print ("15:")
    print (hidden.shape)

    conv = tf.nn.conv2d(hidden, args[30], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + args[31])
    print ("16:")
    print (hidden.shape)

    reshape = tf.reshape(hidden, [-1, 14*14*1000])

    if is_training: hidden = tf.nn.dropout(hidden, 14/128.0)
    hidden = tf.nn.relu(tf.matmul(reshape, args[32]) + args[33])

    return tf.matmul(hidden, args[34]) + args[35]
