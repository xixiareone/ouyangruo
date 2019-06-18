import numpy as np
import tensorflow as tf
from net3 import model
import sys,os
import cPickle as pickle
from math import exp, log
from scipy import misc
from random import randint
import json
from PIL import Image
import matplotlib.pyplot as plt
from del_data import get_testdata

print len(os.listdir('/home/ouyangruo/Documents/BiShe/data/picture_test'))

for name in os.listdir('/home/ouyangruo/Documents/BiShe/data/picture_true'):
    img = np.array(Image.open("/home/ouyangruo/Documents/BiShe/data/picture_true/" + name).convert("L"))
    ss=get_testdata(img)
    print ss
    
            