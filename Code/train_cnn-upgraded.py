import numpy as np
import tensorflow as tf
import sys,os
import cPickle as pickle

with open('/home/ouyangruo/Documents/BiShe/Picture/data0.pickle', 'rb') as m0, open('/home/ouyangruo/Documents/BiShe/Picture/data1.pickle', 'rb') as m1,\
    open('/home/ouyangruo/Documents/BiShe/Picture/data2.pickle', 'rb') as m2, open('/home/ouyangruo/Documents/BiShe/Picture/data3.pickle', 'rb') as m3, \
    open('/home/ouyangruo/Documents/BiShe/Picture/data4.pickle', 'rb') as m4, open('/home/ouyangruo/Documents/BiShe/Picture/data5.pickle', 'rb') as m5, \
    open('/home/ouyangruo/Documents/BiShe/Picture/data6.pickle', 'rb') as m6, open('/home/ouyangruo/Documents/BiShe/Picture/data7.pickle', 'rb') as m7, \
    open('/home/ouyangruo/Documents/BiShe/Picture/data8.pickle', 'rb') as m8, open('/home/ouyangruo/Documents/BiShe/Picture/data9.pickle', 'rb') as m9, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data0.pickle', 'rb') as f0, open('/home/ouyangruo/Documents/BiShe/Picture_one/data1.pickle', 'rb') as f1,\
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data2.pickle', 'rb') as f2, open('/home/ouyangruo/Documents/BiShe/Picture_one/data3.pickle', 'rb') as f3, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data4.pickle', 'rb') as f4, open('/home/ouyangruo/Documents/BiShe/Picture_one/data5.pickle', 'rb') as f5, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data6.pickle', 'rb') as f6, open('/home/ouyangruo/Documents/BiShe/Picture_one/data7.pickle', 'rb') as f7, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data8.pickle', 'rb') as f8, open('/home/ouyangruo/Documents/BiShe/Picture_one/data9.pickle', 'rb') as f9: 
  
    train_datasets = np.concatenate((
        pickle.load(m0), pickle.load(m1), pickle.load(m2), pickle.load(m3), pickle.load(m4),
        pickle.load(m5), pickle.load(m6), pickle.load(m7), pickle.load(m8), pickle.load(m9)
    ))
    test_datasets = np.concatenate((
        pickle.load(f0), pickle.load(f1), pickle.load(f2), pickle.load(f3), pickle.load(f4),
        pickle.load(f5), pickle.load(f6), pickle.load(f7), pickle.load(f8), pickle.load(f9)
    ))
    print('Training set', train_datasets.shape)
    print('Test set', test_datasets.shape)


train_size, test_size = 7000, 3000
image_size = 25
num_labels = 10
num_channels = 1 

def make_arrays(nb_rows, img_size):
    if nb_rows:
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        labels = None, None
    return labels


def merge_datasets(pickle_files, train_size):
    num_classes = len(pickle_files)
    train_labels = make_arrays(train_size, image_size)
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_t= tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class

    return train_labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_labels = labels[permutation]
    return shuffled_labels



train_labels = merge_datasets(train_datasets, train_size)
test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_datasets.shape, train_labels.shape)
print('Testing:', test_datasets.shape, test_labels.shape)

train_labels = randomize(train_datasets, train_labels)
test_labels = randomize(test_datasets, test_labels)


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_datasets, train_labels = reformat(train_datasets, train_labels) 
test_datasets, test_labels = reformat(test_datasets, test_labels)
print('Training set', train_datasets.shape, train_labels.shape)
print('Test set', test_datasets.shape, test_labels.shape)


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
  tf_test_dataset = tf.constant(test_datasets)

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
        batch_data = train_datasets[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        session.run([optimizer], feed_dict=feed_dict)
        if (step % 100 == 0):
          l, predictions, l2 = session.run([loss, train_prediction, l2_loss], feed_dict=feed_dict)
          saver.save(session, '/home/ouyangruo/Documents/BiShe/model.ckpt')
          print(learning_rate.eval())
          print('Minibatch loss and l2 loss at step %d: %f, %f' % (step, l, l2))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
          
    