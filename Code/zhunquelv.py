import numpy as np
import tensorflow as tf
import sys,os
import cPickle as pickle
import matplotlib.pyplot as plt
from net3 import graph, model, get_args
train_size, test_size = 666, 267
image_size = 25
num_labels = 10
num_channels = 1 

model_path = '/home/ouyangruo/Documents/BiShe/Model_one'


with open('/home/ouyangruo/Documents/BiShe/Picture_one/data0.pickle', 'rb') as f0, open('/home/ouyangruo/Documents/BiShe/Picture_one/data1.pickle', 'rb') as f1,\
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data2.pickle', 'rb') as f2, open('/home/ouyangruo/Documents/BiShe/Picture_one/data3.pickle', 'rb') as f3, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data4.pickle', 'rb') as f4, open('/home/ouyangruo/Documents/BiShe/Picture_one/data5.pickle', 'rb') as f5, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data6.pickle', 'rb') as f6, open('/home/ouyangruo/Documents/BiShe/Picture_one/data7.pickle', 'rb') as f7, \
    open('/home/ouyangruo/Documents/BiShe/Picture_one/data8.pickle', 'rb') as f8, open('/home/ouyangruo/Documents/BiShe/Picture_one/data9.pickle', 'rb') as f9: 
    
    test_dataset = np.concatenate((
        pickle.load(f0), pickle.load(f1), pickle.load(f2), pickle.load(f3), pickle.load(f4),
        pickle.load(f5), pickle.load(f6), pickle.load(f7), pickle.load(f8), pickle.load(f9)
    ))
    print('Training set', train_dataset.shape)
    print('Test set', test_dataset.shape)




test_labels = np.zeros([2576, 10], dtype='float32')
test_labels[0:test_size, 0] = 1
test_labels[test_size:test_size+252, 1] = 1
test_labels[test_size+252:test_size+252+271, 2] = 1
test_labels[test_size+252+271:test_size+252+271+273, 3] = 1
test_labels[test_size+252+271+273:test_size+252+271+273+283, 4] = 1
test_labels[test_size+252+271+273+283:test_size+252+271+273+283+233, 5] = 1
test_labels[test_size+252+271+273+283+233:test_size+252+271+273+283+233+236, 6] = 1
test_labels[test_size+252+271+273+283+233+236:test_size+252+271+273+283+233+236+266, 7] = 1
test_labels[test_size+252+271+273+283+233+236+266:test_size+252+271+273+283+233+236+266+238, 8] = 1
test_labels[test_size+252+271+273+283+233+236+266+238:test_size+252+271+273+283+233+236+266+238+257, 9] = 1

test_dataset = test_dataset.reshape(-1, image_size, image_size, 1)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with open('/home/ouyangruo/Documents/BiShe/Model_one/args.pickle','rb') as f:
        args=pickle.loads(f.read())
        session = tf.InteractiveSession()
        
        img=test_dataset[100].reshape(25,25)  
        plt.imshow(img)
        plt.show()
        print np.argmax(test_labels[100])
        
        test_prediction = model(test_dataset, args, False)
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    