import os, sys
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf

# module
from Utils import _variable_on_cpu, _variable_with_weight_decay
from NN_base.nn_skeleton import Autoencoder


class VGG16_Partial_SegNet(Autoencoder):
  def __init__(self, mc):
    Autoencoder.__init__(self, mc)

  def _loss(self, labels):
    """Weighted loss graph"""
    return self._weight_loss(self.logits, labels)

  def build(self):
    # Input normalize
    norm1 = tf.nn.lrn(self.images_node, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

    # Encoder
    with tf.variable_scope('encoder') as scope:
      conv1_1 = self._conv_layer(norm1, [3, 3, 3, 64], name="conv1_1")
      conv1_2 = self._conv_layer(conv1_1, [3, 3, 64, 64], name="conv1_2")
      pool1, pool1_indices = self._max_pool_arg(conv1_2, 2, 2, name='pool1') 

      conv2_1 = self._conv_layer(pool1, [3, 3, 64, 128], name="conv2_1")
      conv2_2 = self._conv_layer(conv2_1, [3, 3, 128, 128], name="conv2_2")
      pool2, pool2_indices = self._max_pool_arg(conv2_2, 2, 2, name='pool2') 

      conv3_1 = self._conv_layer(pool2, [3, 3, 128, 256], name="conv3_1")
      conv3_2 = self._conv_layer(conv3_1, [3, 3, 256, 256], name="conv3_2")
      conv3_3 = self._conv_layer(conv3_2, [3, 3, 256, 256], name="conv3_3")
      pool3, pool3_indices = self._max_pool_arg(conv3_3, 2, 2, name='pool3') 

    # Decoder
    with tf.variable_scope('decoder') as scope:
      up3 = self._deconv_layer(pool3, [2, 2, 256, 256], [self.batch_size, 90, 120, 256], 2, "up3")
      conv_decode3_3 = self._conv_layer(up3, [3, 3, 256, 256], False, name="conv_decode3_3")
      conv_decode3_2 = self._conv_layer(conv_decode3_3, [3, 3, 256, 256], False, name="conv_decode3_2")
      conv_decode3_1 = self._conv_layer(conv_decode3_2, [3, 3, 256, 128], False, name="conv_decode3_1")

      up2 = self._deconv_layer(conv_decode3_1, [2, 2, 128, 128], [self.batch_size, 180, 240, 128], 2, "up2")
      conv_decode2_2 = self._conv_layer(up2, [3, 3, 128, 128], False, name="conv_decode2_2")
      conv_decode2_1 = self._conv_layer(conv_decode2_2, [3, 3, 128, 64], False, name="conv_decode2_1")

      up1 = self._deconv_layer(conv_decode2_1, [2, 2, 64, 64], [self.batch_size, 360, 480, 64], 2, "up1")
      conv_decode1_2 = self._conv_layer(up1, [3, 3, 64, 64], False, name="conv_decode1_2")
      conv_decode1_1 = self._conv_layer(conv_decode1_2, [3, 3, 64, 64], False, name="conv_decode1_1")

    # Classification
    # output predicted class number 
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, self.n_classes],
                                           initializer=self._msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1_1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [self.n_classes], tf.constant_initializer(0.0))
      self.logits = tf.nn.bias_add(conv, biases, name=scope.name) # conv classifier logits

    # Total loss
    self.total_loss = self._loss(self.labels)
