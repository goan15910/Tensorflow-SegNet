import os, sys
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf

# module
from Utils import _variable_on_cpu, _variable_with_weight_decay
from NN_base.nn_skeleton import Autoencoder


class Simple_SegNet(Autoencoder):
  def __init__(self, mc):
    Autoencoder.__init__(self, mc)

  def _loss(self, labels):
    """Weighted loss graph"""
    return self._weight_loss(self.logits, labels)

  def build(self):
    # Input normalize
    with tf.variable_scope('input_norm') as scope:
      norm1 = tf.nn.lrn(self.images_node, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

    # Encoder
    with tf.variable_scope('encoder') as scope:
      conv1 = self._conv_layer(norm1, [7, 7, 3, 64], name="conv1")
      pool1, pool1_indices = self._max_pool_arg(conv1, 2, 2, name='pool1') 
      
      conv2 = self._conv_layer(pool1, [7, 7, 64, 64], name="conv2")
      pool2, pool2_indices = self._max_pool_arg(conv2, 2, 2, name='pool2') 
      
      conv3 = self._conv_layer(pool2, [7, 7, 64, 64], name="conv3")
      pool3, pool3_indices = self._max_pool_arg(conv3, 2, 2, name='pool3') 
      
      conv4 = self._conv_layer(pool3, [7, 7, 64, 64], name="conv4")
      pool4, pool4_indices = self._max_pool_arg(conv4, 2, 2, name='pool4')

    # Decoder
    with tf.variable_scope('decoder') as scope:
      upsample4 = self._deconv_layer(pool4, [2, 2, 64, 64], [self.batch_size, 45, 60, 64], 2, "up4")
      conv_decode4 = self._conv_layer(upsample4, [7, 7, 64, 64], act=False, name="conv_decode4")
    
      upsample3= self._deconv_layer(conv_decode4, [2, 2, 64, 64], [self.batch_size, 90, 120, 64], 2, "up3")
      conv_decode3 = self._conv_layer(upsample3, [7, 7, 64, 64], act=False, name="conv_decode3")
    
      upsample2= self._deconv_layer(conv_decode3, [2, 2, 64, 64], [self.batch_size, 180, 240, 64], 2, "up2")
      conv_decode2 = self._conv_layer(upsample2, [7, 7, 64, 64], act=False, name="conv_decode2")
    
      upsample1= self._deconv_layer(conv_decode2, [2, 2, 64, 64], [self.batch_size, 360, 480, 64], 2, "up1")
      conv_decode1 = self._conv_layer(upsample1, [7, 7, 64, 64], act=False, name="conv_decode1")
    
    # Classification
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, self.n_classes],
                                           initializer=self._msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [self.n_classes], tf.constant_initializer(0.0))
      self.logits = tf.nn.bias_add(conv, biases, name=scope.name) # conv classifier logits

    # Total loss
    self.total_loss = self._loss(self.labels)
