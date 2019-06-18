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
    
    test_dataset0 = np.concatenate((
        pickle.load(f0)
    ))
    test_dataset1=np.concatenate((
        pickle.load(f1)
    ))
    test_dataset2=np.concatenate((
        pickle.load(f2)
    ))
    test_dataset3=np.concatenate((
        pickle.load(f3)
    ))
    test_dataset4=np.concatenate((
        pickle.load(f4)
    ))
    test_dataset5=np.concatenate((
        pickle.load(f5)
    ))
    test_dataset6=np.concatenate((
        pickle.load(f6)
    ))
    test_dataset7=np.concatenate((
        pickle.load(f7)
    ))
    test_dataset8=np.concatenate((
        pickle.load(f8)
    ))
    test_dataset9=np.concatenate((
        pickle.load(f9)
    ))
    
test_dataset0 = test_dataset0.reshape(-1, image_size, image_size, 1)    
test_dataset1 = test_dataset1.reshape(-1, image_size, image_size, 1)
test_dataset2 = test_dataset2.reshape(-1, image_size, image_size, 1)
test_dataset3 = test_dataset3.reshape(-1, image_size, image_size, 1)
test_dataset4= test_dataset4.reshape(-1, image_size, image_size, 1)
test_dataset5 = test_dataset5.reshape(-1, image_size, image_size, 1)
test_dataset6 = test_dataset6.reshape(-1, image_size, image_size, 1)
test_dataset7 = test_dataset7.reshape(-1, image_size, image_size, 1)
test_dataset8 = test_dataset8.reshape(-1, image_size, image_size, 1)
test_dataset9 = test_dataset9.reshape(-1, image_size, image_size, 1)


test_label0 = np.zeros([267, 10], dtype='float32')
test_label0[0:test_size, 0] = 1
test_label1 = np.zeros([252,10],  dtype='float32')
test_label1[0:252,1]=1
test_label2 = np.zeros([271,10],  dtype='float32')
test_label2[0:271,2]=1
test_label3 = np.zeros([273,10],  dtype='float32')
test_label3[0:273,3]=1
test_label4 = np.zeros([283,10],  dtype='float32')
test_label4[0:283,4]=1
test_label5 = np.zeros([233,10],  dtype='float32')
test_label5[0:233,5]=1
test_label6 = np.zeros([236,10],  dtype='float32')
test_label6[0:236,6]=1
test_label7 = np.zeros([266,10],  dtype='float32')
test_label7[0:266,7]=1
test_label8 = np.zeros([238,10],  dtype='float32')
test_label8[0:238,8]=1
test_label9 = np.zeros([257,10],  dtype='float32')
test_label9[0:257,9]=1

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with open('/home/ouyangruo/Documents/BiShe/Model_one/args.pickle','rb') as f:
        args=pickle.loads(f.read())
        session = tf.InteractiveSession()
        
        test_prediction0 = model(test_dataset0, args, False)
        test_prediction1 = model(test_dataset1, args, False)
        test_prediction2 = model(test_dataset2, args, False)
        test_prediction3 = model(test_dataset3, args, False)
        test_prediction4 = model(test_dataset4, args, False)
        test_prediction5 = model(test_dataset5, args, False)
        test_prediction6 = model(test_dataset6, args, False)
        test_prediction7 = model(test_dataset7, args, False)
        test_prediction8 = model(test_dataset8, args, False)
        test_prediction9 = model(test_dataset9, args, False) 
        print('Test accuracy: %.1f%%' % accuracy(test_prediction0.eval(), test_label0))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction1.eval(), test_label1))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction2.eval(), test_label2))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction3.eval(), test_label3))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction4.eval(), test_label4))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction5.eval(), test_label5))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction6.eval(), test_label6))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction7.eval(), test_label7))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction8.eval(), test_label8))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction9.eval(), test_label9))
        
    