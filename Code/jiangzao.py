#!/usr/bin/env python
# vim: set fileencoding=utf-8:

#https://my.oschina.net/jhao104/blog/647326?fromerr=xJxwPW5X
 
import sys,os
from PIL import Image,ImageDraw
def saveImage(filename,size):

    image.save(filename)


for name in os.listdir('/home/ouyangruo/Documents/BiShe/data/data11'):
    image = Image.open("/home/ouyangruo/Documents/BiShe/data/data11/" + name).convert("L")
    #image.show()
    
    # 二值化是图像分割的一种常用方法。在二值化图象的时候把大于某个临界灰度值的像素灰度
    #设为灰度极大值，把小于这个值的像素灰度设为灰度极小值，从而实现二值化（一般设置为0-1）。
    #根据阈值选取的不同，二值化的算法分为固定阈值和自适应阈值，这里选用比较简单的固定阈值。

    #把像素点大于阈值的设置,1，小于阈值的设置为0。生成一张查找表，再调用point()进行映射。

    threshold = 140
    table = []
    for i in range(256):
      if i < threshold:
        table.append(0)
      else:
        table.append(1)
    image = image.point(table, '1')
    saveImage("/home/ouyangruo/Documents/BiShe/picture110/" + name + ".jpg",image.size)

