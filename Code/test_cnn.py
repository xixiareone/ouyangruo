import numpy as np
import tensorflow as tf
import sys,os
import cPickle as pickle
from math import exp, log
from random import randint
from PIL import Image
import matplotlib.pyplot as plt


image_size = 25
num_labels = 10
num_channels = 1 

def reformat(dataset):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

w,b=[],[]

def read_data(filename):
    with open(filename) as f:
        for line in f.readlines():
            data= [theta_x for theta_x in line.strip()[1:-1]]
            #data=np.array(data)
            data=tf.constant(reformat(data))
    
            
    return data

layer0_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer0_w.txt')
layer0_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer0_b.txt')
layer1_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer1_w.txt')
layer1_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer1_b.txt')
layer2_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer2_w.txt')
layer2_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer2_b.txt')
layer3_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer3_w.txt')
layer3_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer3_b.txt')
layer4_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer4_w.txt')
layer4_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer4_b.txt')
layer5_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer5_w.txt')
layer5_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer5_b.txt')
layer6_weights=read_data('/home/ouyangruo/Documents/BiShe/Code/data_w/layer6_w.txt')
layer6_biases=read_data('/home/ouyangruo/Documents/BiShe/Code/data_b/layer6_b.txt')




def accuracy(predictions, labels):
  #print ("result1:")
  #print np.argmax(predictions,1)
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

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


for name in os.listdir('/home/ouyangruo/Documents/BiShe/data/picture_test'):
    img = np.array(Image.open("/home/ouyangruo/Documents/BiShe/data/picture_test/" + name).convert("L"))
    img = np.array(img) < 150
    a = np.zeros([img.shape[0]+2, img.shape[1]+2], dtype='float32') 
    a[1:-1, 1:-1] = img
    img=a
    #plt.imshow(img)
    #plt.show()
    
    
    h,w = img.shape
    for x in range(1,h-1):
        for y in range(1,w-1):
            surround_4 = img[x,y+1] + img[x-1, y] + img[x+1, y] + img[x,y-1]
            surround_4_x = img[x-1,y+1] + img[x+1,y+1] + img[x-1,y-1] + img[x+1,y-1]
            surround_8 = surround_4 + surround_4_x
            if img[x,y] and (surround_4 <= 1 or surround_8 <= 2):
                img[x,y] = 0
    
    
    px=img.sum(0)
    i = 1
    j = px.shape[0] - 1
    while(i < j and px[i] <= 1): i+=1
    while(i < j and px[j] <= 1): j-=1
    img = img[:,i:j+1]
    
    
    px = img.sum(0)
    d = img.shape[1] / 4
    p1, p2, p3 = int(d), int(d * 2), int(d * 3)
    def fine_tune(start):
        for i in range(3):
            new_start = px[start-5:start+5].argmin() + start-5
            if start == new_start: 
                end = start
                while(end < w and px[start] == px[end]): end += 1
                return (start + end) >> 1
            start = new_start
        return start
    p1, p2, p3 = fine_tune(p1), fine_tune(p2), fine_tune(p3)
    
    #img[:, p1] = img[:, p2] = img[:, p3] = 0.5
    #plt.imshow(img)
    #plt.show()

    
    child_img_list = []
    def crop(start, end):
        a = np.zeros([25, 25])  
        length = end - start
        edge = (25 - length) >> 1
        a[:,edge:edge+length] = img[1:-1, start:start+length]
        return a
    child_img=crop(0,p1)
    child_img_list.append(child_img)
    child_img=crop(p1,p2)
    child_img_list.append(child_img)
    child_img=crop(p2,p3)
    child_img_list.append(child_img)
    child_img=crop(p3,img.shape[1])
    child_img_list.append(child_img)
    img=child_img_list
    #img=np.array(img)
    
    for i in range(4):
        plt.imshow(img[i])
        plt.show()
        test_dataset=reformat(img[i])
        tf_test_dataset = tf.constant(test_dataset)
        logits = model(tf_test_dataset)
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
        print ("result:")
        print test_prediction