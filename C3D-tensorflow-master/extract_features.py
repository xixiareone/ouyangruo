#!/usr/bin/env python

"""Trains and Evaluates the 3d convolutional neural network using a feed 
    dictionary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import re
import time
import h5py


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np
import input_data
import c3d_model
from PIL import Image

def placeholder_inputs():
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  return images_placeholder

def tower_feature(scope, images):
  """Calculate the total loss and accuracy on a single tower running the model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'
    images: input images with shape 
      [batch_size, sequence_length, height, width, channel]
    labels: label ground truth
      [batch_size]

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """  
  # Build the inference Graph
  with tf.variable_scope("c3d_var") as c3d_scope:
    try:
      logits = c3d_model.inference_c3d(images)
    except ValueError:
      c3d_scope.reuse_variables()
      logits = c3d_model.inference_c3d(images)
  return logits


def run_testing():
  with tf.Graph().as_default():
    # Get the image and the labels placeholder
    images_placeholder = placeholder_inputs()

    with tf.name_scope('%s' % (c3d_model.TOWER_NAME)) as scope:
      # Calculate the loss and accuracy for one tower for the model. This 
      # function constructs the entire model but shares the variables 
      # across all towers.
      features = tower_feature(scope, images_placeholder)

    # Create a saver
    saver = tf.train.Saver(tf.global_variables())

    # Build an initialization operation to run below
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True))

    # Retore the training model from check point

    if os.path.isfile('pretrained_model/sports1m_finetuning_ucf101.model'):
      print("model exist")
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        sess.run(init)
        # Variable to restore
        variables = {
          "var_name/wc1": tf.get_variable('c3d_var/conv1/weight'),
          "var_name/wc2": tf.get_variable('c3d_var/conv2/weight'),
          "var_name/wc3a": tf.get_variable('c3d_var/conv3/weight_a'),
          "var_name/wc3b": tf.get_variable('c3d_var/conv3/weight_b'),
          "var_name/wc4a": tf.get_variable('c3d_var/conv4/weight_a'),
          "var_name/wc4b": tf.get_variable('c3d_var/conv4/weight_b'),
          "var_name/wc5a": tf.get_variable('c3d_var/conv5/weight_a'),
          "var_name/wc5b": tf.get_variable('c3d_var/conv5/weight_b'),
          "var_name/wd1": tf.get_variable('c3d_var/local6/weights'),
          #"var_name/wd2": tf.get_variable('c3d_var/local7/weights'),
          "var_name/bc1": tf.get_variable('c3d_var/conv1/biases'),
          "var_name/bc2": tf.get_variable('c3d_var/conv2/biases'),
          "var_name/bc3a": tf.get_variable('c3d_var/conv3/biases_a'),
          "var_name/bc3b": tf.get_variable('c3d_var/conv3/biases_b'),
          "var_name/bc4a": tf.get_variable('c3d_var/conv4/biases_a'),
          "var_name/bc4b": tf.get_variable('c3d_var/conv4/biases_b'),
          "var_name/bc5a": tf.get_variable('c3d_var/conv5/biases_a'),
          "var_name/bc5b": tf.get_variable('c3d_var/conv5/biases_b'),
          "var_name/bd1": tf.get_variable('c3d_var/local6/biases'),
          #"var_name/bd2": tf.get_variable('c3d_var/local7/biases')
        }
        saver_c3d = tf.train.Saver(variables)
        saver_c3d.restore(sess, 'pretrained_model/sports1m_finetuning_ucf101.model')
    else:
        print('cannot load model')

    with open('list/test.list', 'r') as f:
      lines = f.readlines()
    h5_file = h5py.File('D:/Python/Python_codes/C3D-tensorflow-master/test/test_4096.h5','w')
    for line in lines:
      print(line)
      test_sample_path, label = line.strip().split(' ')
      img_arr = []
      for filename in os.listdir(test_sample_path):
          img = Image.open(os.path.join(test_sample_path, filename)).resize((c3d_model.CROP_SIZE, c3d_model.CROP_SIZE), Image.ANTIALIAS)
          print(filename)
          # (112, 112, 3) -> (1, 112, 112, 3)
          img_arr.append(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
      N = len(img_arr)
      img_features = []
      for i in range(N-15):
          img_data = np.expand_dims(np.concatenate(img_arr[i:i+16], axis=0), axis=0) # Duixiang is for list
          img_features.append(sess.run(features, feed_dict={images_placeholder: img_data}))

      img_features = np.concatenate(img_features, axis=0) if len(img_features) > 1 else np.array(img_features, dtype=np.float32)
      print(img_features.shape)
      key = label + '_' + test_sample_path.split('/')[-1]
      h5_file[key] = img_features


def main(_):
  # Set the gpu visial device
  os.environ["CUDA_VISIBLE_DEVICES"]='0'
  run_testing()


if __name__ == '__main__':
  tf.app.run()
