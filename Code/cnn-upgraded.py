import numpy as np
import tensorflow as tf
import sys,os
import cPickle as pickle
from net3 import graph, model, get_args, model_path

pickle_file = '/home/ouyangruo/Documents/BiShe/data.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  print ("train_labels")
  print train_labels
  print ("test_labels")
  print test_labels

image_size = 25
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  #print ("result1:")
  #print np.argmax(predictions,1)
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

batch_size = 128

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  layer0_weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
  layer0_biases = tf.Variable(tf.zeros([32]))

  layer1_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([64]))

  layer2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[64]))

  layer3_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 32], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[32]))

  layer4_weights = tf.Variable(tf.truncated_normal([4 * 4 * 32, 64], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[64]))

  layer5_weights = tf.Variable(tf.truncated_normal([64, 64], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[64]))

  layer6_weights = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1))
  layer6_biases = tf.Variable(tf.constant(1.0, shape=[10]))


  #Model.
  #Now instead of using strides = 2 for convolutions we will use maxpooling with
  #same convolution sizes
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

  # Training computation.
  logits = model(tf_train_dataset)
  global_step = tf.Variable(0)
  start_learning_rate = 0.005
  decay_steps = 1000
  decay_size = 0.95
  learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_size)

  saver = tf.train.Saver()

  beta = 0.0005
  l2_loss = (beta*tf.nn.l2_loss(layer0_weights)
      + beta*tf.nn.l2_loss(layer0_biases)
      + beta*tf.nn.l2_loss(layer1_weights)
      + beta*tf.nn.l2_loss(layer1_biases)
      + beta*tf.nn.l2_loss(layer2_weights)
      + beta*tf.nn.l2_loss(layer2_biases)
      + beta*tf.nn.l2_loss(layer3_weights)
      + beta*tf.nn.l2_loss(layer3_biases)
      + beta*tf.nn.l2_loss(layer4_weights)
      + beta*tf.nn.l2_loss(layer4_biases)
      + beta*tf.nn.l2_loss(layer5_weights)
      + beta*tf.nn.l2_loss(layer5_biases)
      + beta*tf.nn.l2_loss(layer6_weights)
      + beta*tf.nn.l2_loss(layer6_biases))

  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels) + l2_loss
  )
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 10001

with tf.Session(graph=graph) as session:
    ckpt = tf.train.get_checkpoint_state('model')
    print(ckpt)
    try:
        assert (ckpt and ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    except:
        tf.global_variables_initializer().run()
        print("--------- load error!!! --------")

        
    for step in xrange(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        session.run([optimizer], feed_dict=feed_dict)
        if (step % 100 == 0):
          l, predictions, l2 = session.run([loss, train_prediction, l2_loss], feed_dict=feed_dict)
          saver.save(session, '/ouyangruo/Documents/BiShe/model/model.ckpt')
          print(learning_rate.eval())
          print('Minibatch loss and l2 loss at step %d: %f, %f' % (step, l, l2))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
          print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
          
    