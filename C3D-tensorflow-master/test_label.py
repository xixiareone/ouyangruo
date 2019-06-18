#!/usr/bin/env python
#coding=utf-8


import os
index=0

for name in os.listdir('D:/Python/Python_codes/C3D-tensorflow-master/list/media/6TB/UCF-101'):
	path='D:/Python/Python_codes/C3D-tensorflow-master/list/media/6TB/UCF-101/'+name+'/'
	# print('1',path)    
	f=os.listdir(path)
	n=0 
	for i in f: 
		oldname=path+f[n]
		# print('2',oldname)
		# print('3',f[n])
		# num=len(f[n])
		# for j in range(num):
		# 	if(f[n][j]=='.'):
		# 		index=j
		newname=path+'/'+f[n]+str(' ')+str(index)
		os.rename(oldname,newname) 
		n+=1
	index=index+1