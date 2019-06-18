#!/usr/bin/env python
# vim: set fileencoding=utf-8:

import numpy as np
import sys,os
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt

file_index=[]
def saveImage(filename,img,i):

  misc.imsave(filename,img[i])
  #plt.imshow(img[i])
  #plt.savefig(filename)

def judge(file_name,index_one,file_index,img,i):
    #print("test")
    #print file_index
    if (file_name[index_one] == '0'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data0/" + file_index + ".jpg",img,i)   
    elif (file_name[index_one] == '1'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data1/" + file_index + ".jpg",img,i)   
    elif (file_name[index_one] == '2'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data2/" + file_index + ".jpg",img,i)
    elif (file_name[index_one] == '3'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data3/" + file_index + ".jpg",img,i)
    elif (file_name[index_one] == '4'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data4/" + file_index + ".jpg",img,i)  
    elif (file_name[index_one] == '5'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data5/" + file_index + ".jpg",img,i)   
    elif (file_name[index_one] == '6'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data6/" + file_index + ".jpg",img,i)   
    elif (file_name[index_one] == '7'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data7/" + file_index + ".jpg",img,i)   
    elif (file_name[index_one] == '8'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data8/" + file_index + ".jpg",img,i)   
    elif (file_name[index_one] == '9'):
        saveImage("/home/ouyangruo/Documents/BiShe/Picture_one/data9/" + file_index + ".jpg",img,i)   


for name in os.listdir('/home/ouyangruo/Documents/BiShe/data/data11'):
    img = np.array(Image.open("/home/ouyangruo/Documents/BiShe/data/data11/" + name).convert("L"))
    img = np.array(img) < 150
    a = np.zeros([img.shape[0]+2, img.shape[1]+2], dtype='float32') #图片加了一圈，让图片完全的降噪，就是除掉边界的干扰线
    a[1:-1, 1:-1] = img
    img=a
    #plt.imshow(img)
    #plt.show()
    
    ########### 去噪 #################
    h,w = img.shape
    for x in range(1,h-1):
        for y in range(1,w-1):
            surround_4 = img[x,y+1] + img[x-1, y] + img[x+1, y] + img[x,y-1]
            surround_4_x = img[x-1,y+1] + img[x+1,y+1] + img[x-1,y-1] + img[x+1,y-1]
            surround_8 = surround_4 + surround_4_x
            if img[x,y] and (surround_4 <= 1 or surround_8 <= 2):
                img[x,y] = 0
    
    ########### 切除左右边界 ##########
    px=img.sum(0)
    i = 1
    j = px.shape[0] - 1
    while(i < j and px[i] <= 1): i+=1
    while(i < j and px[j] <= 1): j-=1
    img = img[:,i:j+1]
    
    ########### 调整切割线 ###########
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

    
    ######## 切割 ########
    child_img_list = []
    def crop(start, end):
        a = np.zeros([25, 25])  #切成25×25,放在神经网络
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
    #print name 
    str_list = name.split('chinamobile_')
    file_name=str_list[1]
    #print file_name
    index = file_name.find('_')
    print index
    index1=index+1
    index2=index+2
    index3=index+3
    index4=index+4
    #print file_name[index1]
    #print file_name[index2]
    #print file_name[index3]
    #print file_name[index4]
    for i in range(4):
        #plt.imshow(img[i])
        #plt.show()
        if(i==0):
            file_index=file_name[:index1+1]
            print file_index
            judge(file_name,index1,file_index,img,i)
        elif (i==1):
            file_index=file_name[:index2+1]
            print file_index
            judge(file_name,index2,file_index,img,i)
        elif (i==2):
            file_index=file_name[:index3+1]
            print file_index
            judge(file_name,index3,file_index,img,i)
        elif (i==3):
            file_index=file_name[:index4+1]
            print file_index
            judge(file_name,index4,file_index,img,i)
            