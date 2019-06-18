from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
import random,sys
import numpy as np
import math
import matplotlib.pyplot as plt

#显示图像灰度直方图#
img=np.array(Image.open('demo.jpg').convert('L'))#图像转化为灰度图
plt.figure("灰色直方图")
arr=img.flatten()#图像矩阵转化为一维数组，用于计算直方图
#hist绘制直方图，bins=256表示直方图的柱数为256，normed=1表示将得到的直方图向量进行归一化，facecolor表示颜色为灰色，alpha=0.75表示透明度为0.75
#n是直方图向量,bins返回各个bin的区间范围,patches返回每个bin里面包含的数据，是一个List
n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='gray', alpha=0.75) 
plt.savefig('灰色直方图.jpg')
plt.show()


#线性转化的对比度和亮度
#通过PIL的ImageEnhance模块来对图像快速的进行对比度增强和亮度增强
#亮度增强通过Brightness类的enhance方法实现
#对比度增强通过Contrast类的enhance方法实现
#这些操作返回一个新的Image对象
img = Image.open("demo.jpg").convert('RGB') 
irange=4.0#irange代表图像增强参数的调整范围
i=2
#factor表示图像增强强度参数

#亮度增强        
imgenhancer_Brightness=ImageEnhance.Brightness(img)
factor=i/irange
img_enhance_Brightness=imgenhancer_Brightness.enhance(factor)
img_enhance_Brightness.show(factor)
img_enhance_Brightness.save("Brightness_%.2f.jpg" %factor) 
#对比度增强      
imgenhancer_Contrast=ImageEnhance.Contrast(img)
factor=i/irange
img_enhance_Contrast=imgenhancer_Contrast.enhance(factor)
img_enhance_Contrast.show("Contrast %f" %factor)
img_enhance_Contrast.save("Contrast_%.2f.jpg" %factor) 

img = Image.open('demo.jpg')
img_one = np.array(img)
rows = img_one.shape[0]#图像是一个矩阵,rows是此图像矩阵的行
cals = img_one.shape[1]#cals是此图像的列
b = np.array(img.convert('L'))

# #伽玛变换#
for i in range(rows):
	for j in range(cals):
		b[i,j]=6*pow(b[i,j],0.5)
plt.imshow(b)
plt.savefig('伽玛变换.jpg')
plt.show()

#二值化变换#
#设置一个阈值，当小于阈值的，则为0，当大于等于阈值的，则为255,255和0分别代表黑色和白色，显示黑白分明的图像显示效果
judge_color = 127.5
for i in range(rows):
	for j in range(cals):
		if(b[i,j]<judge_color):
			b[i,j]=0
		else:
			b[i,j]=255
plt.imshow(b)
plt.savefig('二值图像.jpg')
plt.show()

#直方图均衡化#
#步骤：
#计算直方图，获取每个像素值的个数，计算累积分布函数，即Imhist.cumsum来计算像素个数累积占总数的比重
#当灰度值分布概率相同，累积分布函数f(x)=x/255,将现在图像的累积分布函数的每个值映射到分布概率相同时应有的值，也就是x=f(x)*255
#x轴是bins[:256],y轴是cdf
img=np.array(Image.open('demo.jpg').convert('L'))
#获取直方图
imhist,bins,patches = plt.hist(img.flatten(),bins=256,normed=1,facecolor='gray',alpha=0.75)
cdf = imhist.cumsum()#累积分布函数
cdf = cdf*255/cdf[-1]#进行归一化
im2 = np.interp(img.flatten(),bins[:256],cdf)#使用累积分布函数的线性插值，计算新的像素值
im2 = im2.reshape(img.shape)
plt.hist(im2.flatten(),256)
plt.savefig('直方图均衡化.jpg')
plt.show()

# 对图像向左向右平移20个像素，中心旋转30度，缩小为0.5倍
index=20
img = np.array(Image.open('demo.jpg'))
h = img.shape[0]
w = img.shape[1]
im = np.zeros(img.shape,np.uint8)

for i in range(h):
	for j in range(w):
		im[i,j] = img[i-index,j]#向左平移20个像素
img_one = Image.fromarray(im)#数组恢复为图片
img_one = img_one.rotate(30)#中心旋转30度
img_one = img_one.resize((int(h*0.5),int(w*0.5)),Image.ANTIALIAS)#高度和宽度缩小为0.5倍
plt.imshow(img_one)
plt.savefig('demo1.jpg')
plt.show()

#用自定义5*5平均模板完成图像滤波，重复边界填充方式
blur = cv2.blur(img,(5,5))
plt.imshow(blur)
plt.savefig('均值滤波.jpg')
plt.show()

#高斯平滑#
blur = cv2.GaussianBlur(img,(3,3),0)#高斯平滑函数
plt.imshow(blur)
plt.savefig('高斯平滑.jpg')
plt.show()

#椒盐噪声&中值滤波#
rows,cols,dims=img.shape#图像的宽度和高度和颜色通道(R,G,B)
for i in range(15000):#添加椒盐噪声
	x=np.random.randint(0,rows)
	y=np.random.randint(0,cols)
	img[x,y,:]=255
plt.imshow(img)
plt.savefig('椒盐噪声.jpg')
plt.show()
blur = cv2.medianBlur(img,3)#中值滤波函数
plt.imshow(blur)
plt.savefig('中值滤波.jpg')
plt.show()