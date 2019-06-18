from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
import random,sys
import numpy as np
import math
import matplotlib.pyplot as plt

img = np.array(Image.open('demo.png').convert('L'))
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
d = img.shape[1] / 3
p1, p2, p3 = int(d), int(d * 2), int(d * 3)
def fine_tune(start):
    for i in range(2):
        new_start = px[start-5:start+5].argmin() + start-4
        if start == new_start: 
            end = start
            while(end < w and px[start] == px[end]): end += 1
            return (start + end) >> 1
        start = new_start
    return start
p1, p2, p3 = fine_tune(p1), fine_tune(p2), fine_tune(p3)

img[:, p1] = img[:, p2] = 0.7
plt.imshow(img)
plt.savefig('分割线.jpg')
plt.show()

    
######## 切割 ########
child_img_list = []
def crop(start, end):
    a = np.zeros([600, 700]) 
    length = end - start
    edge = (600- length) >> 1
    a[:,edge:edge+length] = img[1:-1, start:start+length]
    return a
child_img=crop(0,p1)
child_img_list.append(child_img)
child_img=crop(p1,p2)
child_img_list.append(child_img)
child_img=crop(p2,img.shape[1])
child_img_list.append(child_img)
img=child_img_list

for i in range(3):
    plt.imshow(img[i])
    plt.savefig('分割'+str(i+1)+'.jpg')
    plt.show()
