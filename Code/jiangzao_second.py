#!/usr/bin/env python
# vim: set fileencoding=utf-8:


import numpy as np
import sys,os
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt

#img = np.array(Image.open("/home/ouyangruo/Documents/BiShe/data/data1/chinamobile_1470193658_8685.jpeg").convert('L'))

def saveImage(filename,size):

    #img.save(filename)
    misc.imsave(filename,img)
    
for name in os.listdir('/home/ouyangruo/Documents/BiShe/data/data8'):
    img = np.array(Image.open("/home/ouyangruo/Documents/BiShe/data/data8/" + name).convert("L"))
    img = (img < 150).astype('float32')
    #plt.imshow(img)
    #plt.show()
    
    #<150是True >150是FALSE
    #生成一个True False矩阵，然后转化为float，就成1,0,成为二值化矩阵
    #图像的二值化，就是将图像上的像素点的灰度值设置为0或255，也就是将整个图像呈现出明显的只有黑和白的视觉效果
    
    
    
    ##########去噪##############
    h,w = img.shape
    img[:2, :] = img[h-2:, :] = 0
    for x in range(1,h-1):
       for y in range(1,w-1):
         surround_4 = img[x,y+1] + img[x-1, y] + img[x+1, y] + img[x,y-1]   #上下左右
         surround_4_x = img[x-1,y+1] + img[x+1,y+1] + img[x-1,y-1] + img[x+1,y-1]  #对角线 
         surround_8 = surround_4 + surround_4_x
         if img[x,y] and (surround_4 <= 1 or surround_8 <= 2):
            img[x,y] = 0   #删除干扰线
            
    saveImage("/home/ouyangruo/Documents/BiShe/Picture/picture8/" + name + ".jpg",img.size)
    #exit()
    #plt.imshow(img)
    #plt.show()