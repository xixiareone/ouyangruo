# import os
# import numpy as np
# import tensorflow as tf
# import h5py
# import math


# timestep = 5
# test_dataset = []
# test_labels = []
# test_labels_num = np.zeros((101,1))

# f1=h5py.File('D:/Python/Python_codes/C3D-tensorflow-master/test/test_4096.h5','r')
# f = h5py.File('D:/Python/Python_codes/C3D-tensorflow-master/test/test_timestep=5111.h5','w')

# #test2-timestep=3

# #71411
# #69307

# for key, value in f1.items():
# 	ss=np.array(value, dtype=np.float32)
# 	length=len(ss)
# 	if(length)>1:
# 		for i in range(length-timestep):
# 			img_data = np.array(ss[i:i+timestep])
# 			test_dataset.append(img_data)
# 			test_labels.append(int(key.split('_')[0])) 

# print(len(test_dataset))
# print(len(test_labels))
# test_dataset = np.array(test_dataset).astype(np.float32)
# test_labels = np.array(test_labels)
# print(test_dataset.shape) 
# print(test_labels.shape)



# # num = 0
# # for i in range(len(test_labels)):
# # 	for j in range(101):
# # 		if(test_labels[i] == j):
# # 			test_labels_num[j] += 1
# # 			break;

# # for i in range(101):
# # 	print(i,test_labels_num[i])

# # with open('label_shape#train.txt','w')as f:
# # 	for i in range(len(test_labels_num)):
# # 		f.write(str(i))
# # 		f.write(' ')
# # 		f.write(str(test_labels_num[i]))
# # 		f.write('\n')
# f['data'] = test_dataset
# f['label'] = test_labels

import os
import numpy as np
import tensorflow as tf
import h5py
import math


timestep = 7
test_dataset = []
test_labels = []
test_labels_num = np.zeros((101,1))

f1=h5py.File('D:/Python/Python_codes/C3D-tensorflow-master/test/train_4096.h5','r')
f = h5py.File('D:/Python/Python_codes/C3D-tensorflow-master/test/train_timestep=7.h5','w')

#test2-timestep=3

#71411
#69307

for key, value in f1.items():
	ss=np.array(value, dtype=np.float32)
	length=len(ss)
	if(length)>1:
		for i in range(length-timestep):
			img_data = np.expand_dims(ss[i:i+timestep], axis=0)
			# img_data = np.array(ss[i:i+timestep])
			test_dataset.append(img_data)
			test_labels.append(int(key.split('_')[0])) 

print(len(test_dataset))
print(len(test_labels))
test_dataset = np.concatenate(test_dataset, axis=0).astype(np.float32)
test_labels = np.array(test_labels)

print(test_dataset.shape) 
print(test_labels.shape)



# num = 0
# for i in range(len(test_labels)):
# 	for j in range(101):
# 		if(test_labels[i] == j):
# 			test_labels_num[j] += 1
# 			break;

# for i in range(101):
# 	print(i,test_labels_num[i])

# with open('label_shape#train.txt','w')as f:
# 	for i in range(len(test_labels_num)):
# 		f.write(str(i))
# 		f.write(' ')
# 		f.write(str(test_labels_num[i]))
# 		f.write('\n')
f['data'] = test_dataset
f['label'] = test_labels
