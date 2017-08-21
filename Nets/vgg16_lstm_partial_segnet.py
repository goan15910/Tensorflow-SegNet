import os, sys
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf

# module
from Utils import _variable_on_cpu, _variable_with_weight_decay
from NN_base.nn_skeleton import Autoencoder
from NN_base.convLSTM import ConvLSTMCell, ConvGRUCell


class VGG16_LSTM_Partial_SegNet(Autoencoder):
  def __init__(self, mc):
    Autoencoder.__init__(self, mc)

    # ConvLSTM parameters setup
    self.lstm_cell = ConvLSTMCell(256, \
                       k_size=3, \
                       batch_size=self.batch_size, \
                       height=45, \
                       width=60, \
                       initializer=self.conv_init())
    self.state = None


  def build(self):
    for step in xrange(self.seq_len):
      # reuse flag for network
      lstm_reuse = True if step > 0 else None

      # Input normalize
      with tf.variable_scope('norm_input', reuse=lstm_reuse) as scope:
        norm1 = tf.nn.lrn(self.images_node[:, step, ...], depth_radius=5, \
                          bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

      # Encoder
      pool3, _ = self._encoder(norm1, reuse=lstm_reuse)

      # ConvLSTM
      with tf.variable_scope('convLSTM', reuse=lstm_reuse) as scope:
        if step == 0:
          self.state = self.lstm_cell.zero_state(self.batch_size)
        conv_lstm_out, self.state = self.lstm_cell(pool3, self.state)

      # Decoder
      conv_decode1_1 = self._decoder(conv_lstm_out, reuse=lstm_reuse)

      # Classification
      with tf.variable_scope('conv_classifier', reuse=lstm_reuse) as scope:
        logits = self._conv_layer(conv_decode1_1,
                                  [1, 1, 64, self.n_classes],
                                  init=self._msra_initializer(1, 64),
                                  act=False,
                                  wd=0.0005,
                                  batch_norm=False,
                                  name=scope.name)

      # Total loss
      self.total_loss += self._loss(logits, self.labels[:, step, ...])

      # Logits seq
      self.logits.append(logits)

    # Transpose from TNHWC to NTHWC
    self.logits = tf.stack(self.logits)
    self.logits = tf.transpose(self.logits, perm=[1, 0, 2, 3, 4])


  def _encoder(self, inputT, reuse):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
      conv1_1 = self._conv_layer(inputT, [3, 3, 3, 64], name="conv1_1")
      conv1_2 = self._conv_layer(conv1_1, [3, 3, 64, 64], name="conv1_2")
      pool1, pool1_indices = self._max_pool_arg(conv1_2, 2, 2, name='pool1') 

      conv2_1 = self._conv_layer(pool1, [3, 3, 64, 128], name="conv2_1")
      conv2_2 = self._conv_layer(conv2_1, [3, 3, 128, 128], name="conv2_2")
      pool2, pool2_indices = self._max_pool_arg(conv2_2, 2, 2, name='pool2') 

      conv3_1 = self._conv_layer(pool2, [3, 3, 128, 256], name="conv3_1")
      conv3_2 = self._conv_layer(conv3_1, [3, 3, 256, 256], name="conv3_2")
      conv3_3 = self._conv_layer(conv3_2, [3, 3, 256, 256], name="conv3_3")
      pool3, pool3_indices = self._max_pool_arg(conv3_3, 2, 2, name='pool3') 
      
      return pool3, pool3_indices


  def _decoder(self, inputT, reuse):
    with tf.variable_scope('decoder', reuse=reuse) as scope:
      up3 = self._deconv_layer(inputT, [2, 2, 256, 256], [self.batch_size, 90, 120, 256], 2, "up3")
      conv_decode3_3 = self._conv_layer(up3, [3, 3, 256, 256], act=False, name="conv_decode3_3")
      conv_decode3_2 = self._conv_layer(conv_decode3_3, [3, 3, 256, 256], act=False, name="conv_decode3_2")
      conv_decode3_1 = self._conv_layer(conv_decode3_2, [3, 3, 256, 128], act=False, name="conv_decode3_1")

      up2 = self._deconv_layer(conv_decode3_1, [2, 2, 128, 128], [self.batch_size, 180, 240, 128], 2, "up2")
      conv_decode2_2 = self._conv_layer(up2, [3, 3, 128, 128], act=False, name="conv_decode2_2")
      conv_decode2_1 = self._conv_layer(conv_decode2_2, [3, 3, 128, 64], act=False, name="conv_decode2_1")

      up1 = self._deconv_layer(conv_decode2_1, [2, 2, 64, 64], [self.batch_size, 360, 480, 64], 2, "up1")
      conv_decode1_2 = self._conv_layer(up1, [3, 3, 64, 64], act=False, name="conv_decode1_2")
      conv_decode1_1 = self._conv_layer(conv_decode1_2, [3, 3, 64, 64], act=False, name="conv_decode1_1")
      return conv_decode1_1
