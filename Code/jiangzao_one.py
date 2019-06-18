#!/usr/bin/env python
# vim: set fileencoding=utf-8:
 
import sys,os
from PIL import Image,ImageDraw

#二值数组
t2val = {}
def twoValue(image,G):

    #range生成一个list对象
    #xrange:生成器，不用开辟一个很大的内存空间，每次调用返回其中的一个值
    #xrange做循环的性能比range好，尤其是返回很大的值，用xrange
    #若返回列表，就用range
    for y in xrange(0,image.size[1]):         
                                              
        for x in xrange(0,image.size[0]):    
            g = image.getpixel((x,y))   #获取RGB值读取内存      
            if g > G:
                t2val[(x,y)] = 1
            else:
                t2val[(x,y)] = 0

 #降噪
 #根据一个点A的RGB的值，与周围的8个点的RGB值比较，设定一个值N（0<N<8）当A的RGB的值与周围8个点的RGB相等数小于N时，
 #此点作为噪点
 #G:Integer 图像二值化阈值
 #N:Integer 降噪声 0<N<8
 #Z Integer 降噪次数
 # 输出
 # 0：降噪成功
 # 1：降噪失败
def clearNoise(image,N,Z):

    for i in xrange(0,Z):
        t2val[(0,0)] = 1
        t2val[(image.size[0] - 1,image.size[1] - 1)] = 1

        for x in xrange(1,image.size[0] - 1):
            for y in xrange(1,image.size[1] - 1):
                nearDots = 0
                L = t2val[(x,y)]
                if L == t2val[(x - 1,y - 1)]:
                    nearDots += 1
                if L == t2val[(x - 1,y)]:
                    nearDots += 1
                if L == t2val[(x- 1,y + 1)]:
                    nearDots += 1
                if L == t2val[(x,y - 1)]:
                    nearDots += 1
                if L == t2val[(x,y + 1)]:
                    nearDots += 1
                if L == t2val[(x + 1,y - 1)]:
                    nearDots += 1
                if L == t2val[(x + 1,y)]:
                    nearDots += 1
                if L == t2val[(x + 1,y + 1)]:
                    nearDots += 1

                if nearDots < N:
                    t2val[(x,y)] = 1

def saveImage(filename,size):
    image = Image.new("1",size)  #新建立一个图像
    draw = ImageDraw.Draw(image) #创建一个绘画图像

    for x in xrange(0,size[0]):
        for y in xrange(0,size[1]):
            draw.point((x,y),t2val[(x,y)]) #二维绘图

    image.save(filename)
    #image.show()


for name in os.listdir('/home/ouyangruo/Documents/BiShe/data/data11'):
    image = Image.open("/home/ouyangruo/Documents/BiShe/data/data11/" + name).convert("L")  
    twoValue(image,100)
    clearNoise(image,2,0)   #降噪的深度
    saveImage("/home/ouyangruo/Documents/BiShe/picture11/" + name + ".jpg",image.size)

