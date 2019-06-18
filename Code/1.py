#!/usr/bin/env python
# vim: set fileencoding=utf-8:
import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt

img = Image.open("/home/ouyangruo/Documents/BiShe/data/data1/chinamobile_1470193601_9114.jpeg").convert('L')
img = np.array(img) < 140
a = np.zeros([img.shape[0]+2, img.shape[1]+2], dtype='float32')
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
    a = np.zeros([16, 16])
    length = end - start 
    edge = (16 - length) >> 1
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
for i in range(4):
    plt.imshow(child_img_list[i])
    plt.show()
# #plt.imshow(crop(0,p1))
# #plt.show()
# misc.imsave('/home/ouyangruo/Documents/BiShe/Picture/1.jpg',crop(0,p1))
# #plt.imshow(crop(p1,p2))
# #plt.show()
# misc.imsave('/home/ouyangruo/Documents/BiShe/Picture/2.jpg',crop(p1,p2))
# #plt.imshow(crop(p2,p3))
# #plt.show()
# misc.imsave('/home/ouyangruo/Documents/BiShe/Picture/3.jpg',crop(p2,p3))
# #plt.imshow(crop(p3,img.shape[1]))
# #plt.show()
# misc.imsave('/home/ouyangruo/Documents/BiShe/Picture/4.jpg',crop(p3,img.shape[1]))