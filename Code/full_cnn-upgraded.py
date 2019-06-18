#!/usr/bin/env python
#-*- coding: utf-8 -*-



from __future__ import print_function

import numpy as np
import tensorflow as tf

from img_pickle import load_pickle



def reformat(dataset, labels, image_size, num_labels): #将Data降维成一维，将label映射为one-hot encoding
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]  #y的标签
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]



#梯度训练
def tf_logist():
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    #在进行梯度下降训练，即使有些数据不能用，是禁止的数据，但是训练数据的子集跑的会更快
    train_subset = 1000  #为了快速查看训练效果，每论训练只给10000个训练数据（subset）每次都是相同的训练数据


#使用梯度计算train_loss,用tf.Graph()创建一个计算单元
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        #将训练数据，验证数据和测试数据都加载到这个图中
        #tf.constant将dataset和label转化为tensorlfow可用的训练格式（训练中不可修改）
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        #variables这些是将要训练的参数
        #tf.truncateds_normal生成正太分布的数据，作为W的初始值，初始化b为可变的0矩阵
        #tf.variable将上面的矩阵转化为tensoflow可用的训练格式（训练中可以修改） 
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training example: that's our loss.
        #我们将把输入矩阵和权重矩阵相乘，并且添加偏差，我们计算Softmax和Crossrntropy（它是tensorflow中的一种）
        #因为它很常见&可以优化，然后在这些训练数据中取出这些交叉熵的平均值作为损失loss
        #tf.matmul实现矩阵相乘，计算WX+b，这里实际上logit只是一个变量，而非结果
        #tf.nn.sotfmax_cross_entropy_with_logits计算WX+b的结果相较于原来的label的train_loss，并求均值
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        #使用最小梯度来找到train_loss
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        #关于训练，验证和测试数据的预测，这些不是数据的一部分，但是以便我们可以得到训练的准确性如何
        #计算相对的valid_dataset和test_dataset对应的label的train_loss
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    num_steps = 801 #重复计算单元反复训练800多次。提高其准确度
    
    
    #上面这些变量都是一种Tensor的概念，它们是一个个的计算单元
    #我们在Graph中设置了这些计算单元，规定了它们的组合方式，就好像把一个个门电路串起来那样
    
    
    #Session用来执行Graph里规定的计算，就好像给一个个门电路通上电，我们在Session里，给计算单元冲上数据
    
    
    with tf.Session(graph=graph) as session:#将计算单元graph传给session，传给session优化器，train_loss的梯度optimizer，
                                            #训练损失 - train_loss，每次的预测结果，循环执行训练
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        #这是一次性操作，确保图中的1.矩阵的随机权重2.矩阵的零偏差这两个参数进行初始化
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps): #在循环过程中，W和b会保留，并不断的修正
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if step % 100 == 0: 
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels[:train_subset, :]))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                #在valid_prediction上调用.eval（）基本上就像调用run（），但是只是为了得到一个numpy数组，请注意
                #它重新计算所有图的依赖关系
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))  #在每100次循环后，会用验证集进行验证一次，验证也同时修正了一部分参数
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels)) #用测试集来测试


#注意如果lesson 1中没有对数据进行乱序化，可能训练集预测准确度很高，验证集和测试集准确度会很低


def tf_sgd():
    batch_size = 128 

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        #对于训练数据，我们使用一个占位符，将在运行时使用minibatch。
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation. 计算训练
        #每次只取一小部分数据做训练，计算loss时，也只取一小部分数据计算loss
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            #每次输入的训练数据只有128个，随机取起点，取连续128个数据（batch_size=128）
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


#上面SGD的模型只有一层WX+b，现在使用一个RELU作为中间的隐藏层，连接两个WX+b
#仍然只需要修改Graph计算单元为:
#Y = W2 * RELU(W1*X + b1) + b2


def tf_sgd_relu_nn():
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        #由于这里的数据是会变化的，因此用tf.placeholder来存放这块空间
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 1024  #这里N取1024，即1024个隐藏结点(隐藏层的所有节点)
        # Variables.
        #于是四个参数都被修改
        weights1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, hidden_node_count]))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))

        weights2 = tf.Variable(
            tf.truncated_normal([hidden_node_count, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        #预测值计算方法
        ys = tf.matmul(tf_train_dataset, weights1) + biases1
        hidden = tf.nn.relu(ys)
        logits = tf.matmul(hidden, weights2) + biases2
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def load_reformat_not_mnist(image_size, num_labels): 
    pickle_file = '/home/ouyangruo/Documents/BiShe/data.pickle'
    save = load_pickle(pickle_file)
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
    train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

if __name__ == '__main__':
    # First reload the data we generated in 1_notmnist.ipynb.
    image_size = 25
    num_labels = 10
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        load_reformat_not_mnist(image_size, num_labels)

    # tf_logist()
    # tf_sgd()
    tf_sgd_relu_nn()