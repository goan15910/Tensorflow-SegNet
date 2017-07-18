import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
import Image
from math import ceil
import skimage
import skimage.io
from easydict import EasyDict as edict

# modules
from Utils import _variable_on_cpu, _variable_with_weight_decay, print_hist_summery, get_hist, per_class_acc, writeImage, generate_pretrained_initializer
from reader import Reader
from Config import set_config
from Nets import nets_table


class Graph_Runner:
  def __init__(self, FLAGS):
    # Setup from cfg
    cfg = set_config(FLAGS.net, FLAGS.dataset)
    self.cfg = cfg
    self.batch_size = cfg.BATCH_SIZE
    self.image_h = cfg.IMAGE_HEIGHT
    self.image_w = cfg.IMAGE_WIDTH
    self.image_c = cfg.IMAGE_DEPTH
    self.max_steps = cfg.MAX_STEPS
    
    # Setup from FLAGS
    self.testing = True if FLAGS.testing else False
    self.log_dir = FLAGS.log_dir
    self.train_dir = FLAGS.image_dir
    self.val_dir = FLAGS.val_dir
    self.test_dir = FLAGS.test_dir
    self.finetune_ckpt = FLAGS.finetune
    self.pretrained_path = FLAGS.pretrained
    self.is_finetune = True if FLAGS.finetune else False
    # should be changed if your model stored by different convention
    self.startstep = 0 if not self.is_finetune else int(FLAGS.finetune.split('-')[-1])
    self.net_name = FLAGS.net
    self.checkpoint_path = os.path.join(self.log_dir, 'model.ckpt')

    # Setup train info format str
    self.train_format_str = ('{}: step {}, loss = {.2f} ({.1f} examples/sec; {.3f} '
                      'sec/batch)')

    
  def training(self):
    """Interface for training the graph"""
    cfg = self.cfg
    with tf.Graph().as_default():
      self._set_run_components()
      if not cfg.USE_LSTM:
        self._train()
      else:
        self._train_seq2seq()
      self._close_runner()
  

  def testing(self):
    #TODO
    raise NotImplementedError


  def _set_run_components(self):
    """Setup components for running the graph"""
    cfg = self.cfg

    # Setup global step
    self.global_step = tf.Variable(0, trainable=False)

    # Setup reader
    train_reader = Reader(cfg, self.train_dir, 'train')
    val_reader = Reader(cfg, self.val_dir, 'val')
    test_reader = Reader(cfg, self.test_dir, 'test')

    # Setup input queues
    if not cfg.USE_LSTM:
      if not self.testing:
        self.train_queue = train_reader.batch_node()
        self.val_queue = val_reader.batch_node()
      else:
        self.test_queue = test_reader.batch_node()
    else:
      if not self.testing:
        self.train_queue = train_reader.seq_batch_node()
        self.val_queue = val_reader.seq_batch_node()
      else:
        self.test_queue = test_reader.seq_batch_node()
      
    # Setup pretrained initializer
    if self.pretrained_path is not None:
      pretrained_npy = np.load(pretrained_path)
      cfg.PRETRAINED_INITIALIZER = generate_pretrained_initializer(pretrained_npy)

    # Setup model graph
    self.model = nets_table[self.net_name](cfg)
    self.model.build()
    self.train_op = self.model.train(self.global_step)

    # Setup saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Setup session
    self.sess = tf.Session()

    # Build init operation
    if (self.is_finetune == True):
      self.saver.restore(self.sess, self.finetune_ckpt)
    else:
      init = tf.global_variables_initializer()
      self.sess.run(init)

    # Setup coordinators and threads for queue runner
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    # Setup summary writer
    self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
    self.summary_op = tf.summary.merge_all()

    # Setup summary placeholders
    self.average_pl = tf.placeholder(tf.float32)
    self.acc_pl = tf.placeholder(tf.float32)
    self.iu_pl = tf.placeholder(tf.float32)
    self.average_summary = tf.summary.scalar("test_average_loss", self.average_pl)
    self.acc_summary = tf.summary.scalar("test_accuracy", self.acc_pl)
    self.iu_summary = tf.summary.scalar("Mean_IU", self.iu_pl)


  def _close_runer(self):
    """House keeping the runner when finishing running"""
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()


  def _optimize(self):
    """Optimize for one forward pass of train graph
       Return corresponding loss and ground truth
    """
    # Get feed batches for train graph
    image_batch, label_batch = self.sess.run(self.train_queue)

    # Setup feed_dict
    if not self.cfg.USE_LSTM:
      feed_dict = {
        self.model.images_node: image_batch,
        self.model.labels_node: label_batch,
        model.phase_train: True
      }
    else:
      feed_dict = {
        self.model.image_seq_node: image_batch,
        self.model.label_seq_node: label_batch,
        model.phase_train: True
      }

    # Setup optimize op list
    op_list = [self.model.total_loss, self.train_op, \
               self.model.logits, self.model.labels]

    # Forward pass and optimize
    loss_value, _, logits, train_labels = self.sess.run(op_list, feed_dict=feed_dict)
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    return (loss_value, logits, train_labels)


  def _forward(self, input_queue, phase_train):
    """One forward pass for the graph"""
    # Get feed batches for train graph
    image_batch, label_batch = self.sess.run(input_queue)

    # Setup feed_dict
    if not self.cfg.USE_LSTM:
      feed_dict = {
        self.model.images_node: image_batch,
        self.model.labels_node: label_batch,
        model.phase_train: phase_train
      }
    else:
      feed_dict = {
        self.model.image_seq_node: image_batch,
        self.model.label_seq_node: label_batch,
        model.phase_train: phase_train
      }

    # Setup op_list
    if phase_train:
      op_list = [self.model.total_loss, self.model.logits, self.model.labels]
    else:
      logits = self.model.logits
      preds = tf.argmax(self.model.logits, dimension=3)
      op_list = [logits, preds]

    # Forward pass
    if phase_train:
      loss_value, dense_pred, labels = self.sess.run(op_list, feed_dict=feed_dict)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      return (loss_value, dense_pred, labels)
    else:
      dense_pred, im_pred = self.sess.run(op_list, feed_dict=feed_dict)
      return (None, dense_pred, im_pred)


  def _add_val_summary(self, avg_val_loss, acc_total, iu):
    # Set summary string
    loss_summary_str = self.sess.run(average_summary, feed_dict={average_pl: avg_val_loss})
    acc_summary_str = self.sess.run(acc_summary, feed_dict={acc_pl: acc_total})
    iu_summary_str = self.sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})

    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)

    # Add summary
    self.summary_writer.add_summary(summary_str, step)
    self.summary_writer.add_summary(loss_summary_str, step)
    self.summary_writer.add_summary(acc_summary_str, step)
    self.summary_writer.add_summary(iu_summary_str, step)


  def _print_train_info(self, step, loss_value, duration):
    fps = self.batch_size / duration
    cost_time = float(duration)
    print self.train_format_str.format(datetime.now(), step, loss_value, fps, cost_time)


  def _train(self):
    """Train ordinary graph"""
    cfg = self.cfg
      
    # Training loop
    start = self.startstep
    end = self.startstep + self.max_steps
    for step in range(start, end):
      # Optimization for one forward pass
      start_time = time.time()
      loss_value, logits, train_labels = self._optimize()
      duration = time.time() - start_time

      # Print train info
      if step % 10 == 0:
        self._print_train_info(step, loss_value, duration)
        per_class_acc(logits, train_labels)

      # Validation
      if step % 100 == 0:
        print "Start validating ..."
        total_val_loss = 0.0
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        for test_step in range(cfg.VAL_ITER):
          val_loss, val_pred, val_labels = self._forward(self.val_queue, phase_train=True)
          total_val_loss += val_loss
          hist += get_hist(val_pred, val_labels)

        avg_val_loss = total_val_loss / cfg.VAL_ITER
        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        self._add_val_summary(avg_val_loss, acc_total, iu)
        print "val_loss: {}".format(avg_val_loss)
        print_hist_summery(hist)
        print "end validating ..."
        
      # Save checkpoint
      if step % 1000 == 0 or (step + 1) == self.max_steps:
        self.saver.save(self.sess, self.checkpoint_path, global_step=step)


  def _train_seq2seq(self):
    """Train graph with seq2seq IO pairs"""
    # TODO
    raise NotImplementedError
