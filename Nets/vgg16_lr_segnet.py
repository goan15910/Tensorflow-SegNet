import os, sys
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf

# module
from Utils import _variable_on_cpu, _variable_with_weight_decay
from NN_base.nn_skeleton import Autoencoder


class VGG16_LR_SegNet(Autoencoder):
  def __init__(self, mc):
    Autoencoder.__init__(self, mc)

    # Downsample the labels
    self.down_ksize = mc.DOWN_SAMPLE_RATIO
    self.down_stride = mc.DOWN_SAMPLE_RATIO
    with tf.variable_scope('label_resize') as scope:
      labels = self._max_pool(tf.to_float(self.labels_node),
                              self.down_ksize,
                              self.down_stride,
                              name='tmp_labels')
      self.labels = tf.cast(labels, tf.int64, name='labels')


  def _loss(self, labels):
    """Weighted loss graph"""
    return self._weight_loss(self.logits, self.labels)

  def build(self):
    # Resize images
    with tf.variable_scope('image_resize') as scope:
      lr_images = self._max_pool(self.images_node, self.down_ksize, self.down_stride, name='lr_images')

    # Input normalize
    with tf.variable_scope('norm_input') as scope:
      norm1 = tf.nn.lrn(lr_images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

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

      conv4_1 = self._conv_layer(pool3, [3, 3, 256, 512], name="conv4_1")
      conv4_2 = self._conv_layer(conv4_1, [3, 3, 512, 512], name="conv4_2")
      conv4_3 = self._conv_layer(conv4_2, [3, 3, 512, 512], name="conv4_3")
      pool4, pool4_indices = self._max_pool_arg(conv4_3, 2, 2, name='pool4') 

      conv5_1 = self._conv_layer(pool4, [3, 3, 512, 512], name="conv5_1")
      conv5_2 = self._conv_layer(conv5_1, [3, 3, 512, 512], name="conv5_2")
      conv5_3 = self._conv_layer(conv5_2, [3, 3, 512, 512], name="conv5_3")
      pool5, pool5_indices = self._max_pool_arg(conv5_3, 2, 2, name='pool5') 

    # Decoder
    with tf.variable_scope('decoder') as scope:
      up5 = self._deconv_layer(pool5, [2, 2, 512, 512], [self.batch_size, 6, 8, 512], 2, "up5")
      conv_decode5_3 = self._conv_layer(up5, [3, 3, 512, 512], False, name="conv_decode5_3")
      conv_decode5_2 = self._conv_layer(conv_decode5_3, [3, 3, 512, 512], False, name="conv_decode5_2")
      conv_decode5_1 = self._conv_layer(conv_decode5_2, [3, 3, 512, 512], False, name="conv_decode5_1")

      up4 = self._deconv_layer(conv_decode5_1, [2, 2, 512, 512], [self.batch_size, 12, 15, 512], 2, "up4")
      conv_decode4_3 = self._conv_layer(up4, [3, 3, 512, 512], False, name="conv_decode4_3")
      conv_decode4_2 = self._conv_layer(conv_decode4_3, [3, 3, 512, 512], False, name="conv_decode4_2")
      conv_decode4_1 = self._conv_layer(conv_decode4_2, [3, 3, 512, 256], False, name="conv_decode4_1")

      up3 = self._deconv_layer(conv_decode4_1, [2, 2, 256, 256], [self.batch_size, 23, 30, 256], 2, "up3")
      conv_decode3_3 = self._conv_layer(up3, [3, 3, 256, 256], False, name="conv_decode3_3")
      conv_decode3_2 = self._conv_layer(conv_decode3_3, [3, 3, 256, 256], False, name="conv_decode3_2")
      conv_decode3_1 = self._conv_layer(conv_decode3_2, [3, 3, 256, 128], False, name="conv_decode3_1")

      up2 = self._deconv_layer(conv_decode3_1, [2, 2, 128, 128], [self.batch_size, 45, 60, 128], 2, "up2")
      conv_decode2_2 = self._conv_layer(up2, [3, 3, 128, 128], False, name="conv_decode2_2")
      conv_decode2_1 = self._conv_layer(conv_decode2_2, [3, 3, 128, 64], False, name="conv_decode2_1")

      up1 = self._deconv_layer(conv_decode2_1, [2, 2, 64, 64], [self.batch_size, 90, 120, 64], 2, "up1")
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
