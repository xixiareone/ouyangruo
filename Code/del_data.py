from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys,os
from net3 import model
import cPickle as pickle


def get_testdata(img):
    img = np.array(img) < 150
    a = np.zeros([img.shape[0]+2, img.shape[1]+2], dtype='float32') 
    a[1:-1, 1:-1] = img
    img=a
    plt.imshow(img)
    plt.show()
    
    
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
    img=np.array(img)
    
    ss=0
    
    for i in range(4):
        test_dataset=np.array(img[i],dtype='float32').reshape(-1,25,25,1)
        with open('/home/ouyangruo/Documents/BiShe/Model_one/args.pickle','rb') as f:
            args=pickle.loads(f.read())
        session = tf.InteractiveSession()
        
        test_prediction = model(test_dataset, args, False)
        result=np.argmax(test_prediction.eval(),1)
        # plt.title(result)
        # plt.imshow(img[i])
        # plt.show()
        
        if(i==0):
            ss=result*1000
        elif (i==1):
            ss=ss+result*100
        elif (i==2):
            ss=ss+result*10
        elif (i==3):
            ss=ss+result
        f.close()
    return ss