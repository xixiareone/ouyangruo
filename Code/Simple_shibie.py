import pytesseract
import Image
import sys,os
import numpy as np
#image = Image.open('/home/ouyangruo/Documents/BiShe/Picture/picture1/picture1chinamobile_1470193586_5213.jpeg.jpg')


for name in os.listdir('/home/ouyangruo/Documents/BiShe/Picture/picture1'):
    image = Image.open("/home/ouyangruo/Documents/BiShe/Picture/picture1/" + name)
    print pytesseract.image_to_string(image)