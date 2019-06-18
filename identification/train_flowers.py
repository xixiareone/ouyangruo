import os
import numpy as np
import tensorflow as tf
import h5py

from darknet import graph, model, get_args, model_path

file=h5py.File('data.hdf5','r')
train_dataset=file['data'][:10200]
test_dataset=file['data'][10200:10881]
train_labels=file['labels'][:10200]
test_labels=file['labels'][10200:10881]
print('Training set', train_dataset.shape)
print('Test set', test_dataset.shape)


image_size = 224
num_channels = 3 
num_labels = 17

#def reformat(dataset, labels):
# dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
# labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
# return dataset, labels
#train_dataset, train_labels = reformat(train_dataset, train_labels)
#test_dataset, test_labels = reformat(test_dataset, test_labels)
def randomize(dataset, labels):
    print (labels.shape[0])
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset.reshape(-1, image_size, image_size, num_channels), shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset = test_dataset.reshape(-1, image_size, image_size, num_channels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

batch_size = 128

args = get_args()
with graph.as_default():

    # Input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 17))
    tf_test_dataset = tf.constant(test_dataset)

    logits = model(tf_train_dataset, args)
    global_step = tf.Variable(0)
    start_learning_rate = 0.05
    decay_steps = 1000
    decay_size = 0.95
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_size)

    l2_loss = 0
    for v in args:
        l2_loss += tf.nn.l2_loss(v)

    beta = 0.0005
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels) + beta*l2_loss
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, args, False))

    saver = tf.train.Saver()

num_steps = 50001

with tf.Session(graph=graph) as session:
    ckpt = tf.train.get_checkpoint_state(model_path)
    print(ckpt)
    try:
        assert (ckpt and ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    except:
        tf.initialize_all_variables().run()
        print("--------- load error!!! --------")

    offset = 0
    offset_max = train_labels.shape[0] - batch_size
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    for step in range(num_steps):
        offset = (offset + batch_size) % offset_max
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        session.run([optimizer], feed_dict=feed_dict)
        if (step % 100 == 0):
            l, predictions, l2 = session.run([loss, train_prediction, l2_loss], feed_dict=feed_dict)
            print(learning_rate.eval())
            print('Minibatch loss and l2 loss at step %d: %f, %f' % (global_step.eval(), l, beta*l2))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
        if step and (step % 500 == 0):
            saver.save(session, model_path + '/model.ckpt')
            print('------------ checkpoint ----------')
