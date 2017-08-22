import tensorflow as tf
from tensorflow.python.framework import ops

import os, sys
import numpy as np
import math
from math import ceil
from tensorflow.python.ops import gen_nn_ops

# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary


class Autoencoder:
  """Base class of encoder-decoder structure"""
  def __init__(self, mc):
    # Configuration setup
    self.n_classes = mc.NUM_CLASSES
    self.loss_weight = mc.LOSS_WEIGHT
    self.init_lr = mc.INITIAL_LEARNING_RATE
    self.moving_avg_decay = mc.MOVING_AVERAGE_DECAY
    self.batch_size = mc.BATCH_SIZE
    self.use_lstm = mc.USE_LSTM
    self.seq_len = mc.SEQUENCE_LENGTH
    if self.use_lstm:
      assert self.seq_len is not None, \
          "To use LSTM, you need to specify sequence length!"
    
    # Shape setup
    N = mc.BATCH_SIZE
    T = mc.SEQUENCE_LENGTH
    H = mc.IMAGE_HEIGHT
    W = mc.IMAGE_WIDTH
    C = mc.IMAGE_DEPTH
    if not self.use_lstm:
      self.images_shape = (N, H, W, C)
      self.labels_shape = (N, H, W, 1)
    else:
      self.images_shape = (N, T, H, W, C)
      self.labels_shape = (N, T, H, W, 1)

    # Graph placeholder setup
    self.images_node = tf.placeholder(
        tf.float32, self.images_shape,
        name='images_node'
    )
    self.labels_node = tf.placeholder(
        tf.int64, self.labels_shape,
        name='labels_node'
    )
    self.labels = self.labels_node
    self.phase_train = tf.placeholder(tf.bool, name='phase_train')
    # TODO: Add reweight_loss pl to graph to enable softmax loss
    # self.reweight_loss = tf.placeholder(tf.bool, name='reweight_loss')

    # Conv initializer setup
    assert mc.CONV_INIT_TYPE in ['msra', 'orthogonal']
    if mc.CONV_INIT_TYPE == 'msra':
      self.conv_init = self._msra_initializer
    elif mc.CONV_INIT_TYPE == 'orthogonal':
      self.conv_init = self._orthogonal_initializer

    # Pretrained weight initializer setup
    self.pretrained_init = mc.PRETRAINED_INITIALIZER

    # Logits
    if not self.use_lstm:
      self.logits = None
    else:
      self.logits = []

    # Total loss
    self.total_loss = 0.0


  def build(self):
    """Add inference graph"""
    raise NotImplementedError


  def _loss(self, logits, labels, weighted=True):
    """Loss graph"""
    if weighted:
      return self._weight_loss(logits, labels)
    else:
      return self._softmax_loss(logits, labels)


  def train(self, global_step):
    """Train graph"""
    lr = self.init_lr # Fix lr
    loss_averages_op = _add_loss_summaries(self.total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(self.total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        self.moving_avg_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op

  
  def _msra_initializer(self, kl, dl):
      """kl for kernel size, dl for filter number"""
      stddev = math.sqrt(2. / (kl**2 * dl))
      return tf.truncated_normal_initializer(stddev=stddev)


  def _orthogonal_initializer(self, scale=1.1):
      """From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120"""
      def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
      return _initializer


  def _softmax_loss(self, logits, labels):
    """Mean cross entropy without re-weighting"""
    # Calculate the average cross entropy loss across the batch.
    logits = tf.reshape(logits, (-1, self.n_classes))
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


  def _weight_loss(self, logits, labels):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
      logits = tf.reshape(logits, (-1, self.n_classes))
      epsilon = tf.constant(value=1e-10)
      logits = logits + epsilon

      # consturct one-hot label array
      labels = tf.cast(labels, tf.int32)
      label_flat = tf.reshape(labels, (-1, 1))

      # should be [batch ,num_classes]
      labels = tf.reshape(tf.one_hot(label_flat, depth=self.n_classes), (-1, self.n_classes))
      softmax = tf.nn.softmax(logits)
      cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), self.loss_weight), reduction_indices=[1])
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection('losses', cross_entropy_mean)
      loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


  def _conv_layer(self, inputT, shape, stride=1, init=None, \
                  act=True, wd=None, batch_norm=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
      if (self.pretrained_init is not None) and (name in self.pretrained_init.keys()):
        print 'Initialize {} with pretrained weight'.format(name)
        conv_init = self.pretrained_init[name]['conv']
        bias_init = self.pretrained_init[name]['bias']
      elif init is None:
        conv_init = self.conv_init()
        bias_init = tf.constant_initializer(0.0)
      else:
        conv_init = init
        bias_init = tf.constant_initializer(0.0)

      kernel = _variable_with_weight_decay('weights', shape=shape, initializer=conv_init, wd=wd)
      conv = tf.nn.conv2d(inputT, kernel, [1, stride, stride, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], bias_init)
      bias = tf.nn.bias_add(conv, biases)
      if batch_norm is True:
        bias = self._batch_norm(bias, self.phase_train)

      if act is True:
        conv_out = tf.nn.relu(bias)
      else:
        conv_out = bias

    return conv_out


  def _deconv_layer(self, inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
      weights = self._get_deconv_filter(f_shape)
      deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                      strides=strides, padding='SAME')
    return deconv


  def _get_deconv_filter(self, f_shape):
    """reference: https://github.com/MarvinTeichmann/tensorflow-fcn"""
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


  def _max_pool_arg(self, inputT, ksize, stride, name):
    return tf.nn.max_pool_with_argmax(inputT, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
           padding='SAME', name=name)


  def _max_pool(self, inputT, ksize, stride, name):
    return tf.nn.max_pool(inputT, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
           padding='SAME', name=name)


  def _batch_norm(self, inputT, is_training):
    return tf.cond(is_training,
            lambda: tf.contrib.layers.batch_norm(inputT, is_training=True, \
                             center=False, updates_collections=None, \
                             scope="batch_norm", reuse=None),
            lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,\
                             updates_collections=None, center=False, \
                             scope="batch_norm", reuse=True))


  def _resize_labels(self, inputT, k_size, stride, name):
    with tf.variable_scope('label_resize') as scope:
      input_shape = inputT.get_shape().as_list()
      N, T, H, W, C = input_shape
      new_H = int(ceil(H / float(stride)))
      new_W = int(ceil(W / float(stride)))
      if len(input_shape) == 4:
        flat_shape = input_shape
        new_shape = (N, new_H, new_W, C)
      elif len(input_shape) == 5:
        flat_shape = (N*T, H, W, C)
        new_shape = (N, T, new_H, new_W, C)
      labels = tf.reshape(inputT, flat_shape)
      labels = self._max_pool(tf.to_float(labels),
                              k_size, stride, 'pooled_labels')
      labels = tf.reshape(labels, new_shape)
      return tf.cast(labels, tf.int64, name=name)


  def _cff_layer(self, lr_T, hr_T, labels):
    """
       Cascade feature fusion unit, 
       Reference: Hengshuang Zhao et al., 
                  https://arxiv.org/abs/1704.08545
    """
    # TODO
    # + upsample low-res inputT
    #   - classification (conv-classifier + softmax)
    #   - dilated conv with BN, (dilation1, 3*3)
    
    # project high-res inputT (1*1 conv with BN)

    # Element-wise addition (low-res + high-res)
    # ReLU
    pass
