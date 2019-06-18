import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.array(Image.open("D:\data\data1\chinamobile_1470193658_8685.jpeg").convert('L'))
img = (img < 150).astype('float32')
plt.imshow(img)
plt.show()

########### 去噪 #################
h,w = img.shape
img[:2, :] = img[h-2:, :] = 0
for x in range(1,h-1):
    for y in range(1,w-1):
        surround_4 = img[x,y+1] + img[x-1, y] + img[x+1, y] + img[x,y-1]
        surround_4_x = img[x-1,y+1] + img[x+1,y+1] + img[x-1,y-1] + img[x+1,y-1]
        surround_8 = surround_4 + surround_4_x
        if img[x,y] and (surround_4 <= 1 or surround_8 <= 2):
            img[x,y] = 0
plt.imshow(img)
plt.show()

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

img[:, p1] = img[:, p2] = img[:, p3] = 0.5
plt.imshow(img)
plt.show()

######## 切割 ########
a = np.zeros([25, 25])
d = p1
m = (25 - d) >> 1
a[:,m:m+d] = img[:, :d]
plt.imshow(a)
plt.show()