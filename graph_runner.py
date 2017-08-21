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
import imageio
import cv2

# modules
from Utils import print_hist_summery, get_hist, per_class_acc, writeImage, generate_pretrained_initializer
from reader import Reader
from Config import set_config
from Nets import nets_table


class Graph_Runner:
  def __init__(self, FLAGS):
    # Setup from cfg
    cfg = set_config(FLAGS.net, FLAGS.dataset)
    self.cfg = cfg
    self.n_classes = cfg.NUM_CLASSES
    self.batch_size = cfg.BATCH_SIZE
    self.seq_len = cfg.SEQUENCE_LENGTH
    self.image_h = cfg.IMAGE_HEIGHT
    self.image_w = cfg.IMAGE_WIDTH
    self.image_c = cfg.IMAGE_DEPTH
    self.max_steps = cfg.MAX_STEPS
    if not cfg.USE_LSTM:
      self.n_imgs = self.batch_size
    else:
      self.n_imgs = self.batch_size * self.seq_len
    self.cmap = cfg.COLORMAP
    self.cnames = cfg.CLASS_NAMES
 
    # Setup path & flags from FLAGS
    self.log_dir = FLAGS.log_dir
    self.train_dir = FLAGS.image_dir
    self.val_dir = FLAGS.val_dir
    self.test_dir = FLAGS.test_dir
    self.finetune_ckpt = FLAGS.finetune
    self.pretrained_path = FLAGS.pretrained
    self.is_finetune = True if FLAGS.finetune else False
    # should be changed if your model stored by different convention
    if not self.is_finetune:
      self.startstep = 0
    else: 
      self.startstep = int(FLAGS.finetune.split('-')[-1])
    self.net_name = FLAGS.net
    self.checkpoint_path = os.path.join(self.log_dir, 'model.ckpt')
    self.test_ckpt = FLAGS.test_ckpt
    self.save_predict = FLAGS.save_predict
    if self.save_predict:
      self.predict_dir = os.path.join(self.log_dir, \
                                      'predictions')
      if not os.path.isdir(self.predict_dir):
        os.mkdir(self.predict_dir)

    
  def run(self, mode):
    """Interface for running the graph"""
    assert mode in ['train', 'test', 'extract'], \
        "Selected running mode {} are not supported".format(mode)
    with tf.Graph().as_default():
      self._set_run_components(mode)
      if mode == 'train':
        self._train()
      elif mode == 'test':
        self._eval(save_predict=self.save_predict)
      elif mode == 'extract':
        self._extract()
      self._close_runner()


  def _set_run_components(self, mode):
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
      if mode == 'train':
        self.train_queue = train_reader.batch_node()
        self.val_queue = val_reader.batch_node()
      elif mode == 'test':
        self.test_queue = test_reader.batch_node()
    else:
      if mode == 'train':
        self.train_queue = train_reader.seq_batch_node()
        self.val_queue = val_reader.seq_batch_node()
      elif mode == 'test':
        self.test_queue = test_reader.seq_batch_node()
      
    # Setup pretrained initializer
    if self.pretrained_path is not None:
      pretrained_npy = np.load(self.pretrained_path)
      self.cfg.PRETRAINED_INITIALIZER = generate_pretrained_initializer(pretrained_npy)

    # Setup model graph
    self.model = nets_table[self.net_name](cfg)
    self.model.build()
    self.train_op = self.model.train(self.global_step)

    # Setup saver
    if mode == 'train':
      self.saver = tf.train.Saver(tf.global_variables())
    elif mode == 'test':
      #var_avgs = tf.train.ExponentialMovingAverage(
      #               cfg.MOVING_AVERAGE_DECAY)
      #restore_vars = var_avgs.variables_to_restore()
      #self.saver = tf.train.Saver(restore_vars)
      self.saver = tf.train.Saver(tf.global_variables())

    # Setup session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    self.sess = tf.Session(config=sess_config)

    # Initialize graph
    if (mode == 'train') and self.is_finetune:
      self.saver.restore(self.sess, self.finetune_ckpt)
    elif (mode == 'train') and (not self.is_finetune):
      init = tf.global_variables_initializer()
      self.sess.run(init)
    elif (mode == 'test'):
      assert self.test_ckpt is not None, \
          "Specify test ckpt to test"
      self.saver.restore(self.sess, self.test_ckpt)

    # Setup coordinators and threads for queue runner
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    # Setup summary writer
    self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
    self.summary_op = tf.summary.merge_all()

    # Setup validation summary graph
    self.avg_loss_pl = tf.placeholder(tf.float32)
    self.acc_total_pl = tf.placeholder(tf.float32)
    self.mean_iu_pl = tf.placeholder(tf.float32)
    self.fps_pl = tf.placeholder(tf.float32)
    self.avg_loss_summary = tf.summary.scalar("val_average_loss", \
                                              self.avg_loss_pl)
    self.acc_total_summary = tf.summary.scalar("val_accuracy", \
                                               self.acc_total_pl)
    self.mean_iu_summary = tf.summary.scalar("val_mean_IU", \
                                             self.mean_iu_pl)
    self.fps_summary = tf.summary.scalar("val_mean_FPS", \
                                         self.fps_pl)


  def _close_runner(self):
    """House keeping the runner when finishing running"""
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()


  def _feed_dict(self, input_queue, phase_train):
    # Get feed batches the graph
    image_batch, label_batch = self.sess.run(input_queue)

    # Setup feed_dict
    feed_dict = {
      self.model.images_node: image_batch,
      self.model.labels_node: label_batch,
      self.model.phase_train: phase_train
    }

    return feed_dict


  def _optimize(self, feed_dict):
    """Optimize for one forward pass of train graph"""
    # Setup optimize op list
    op_list = [self.model.total_loss, self.train_op, \
               self.model.logits, self.model.labels]

    # Forward pass and optimize
    total_loss, _, logits, train_labels = self.sess.run(op_list, feed_dict=feed_dict)
    assert not np.isnan(total_loss), 'Model diverged with loss = NaN'
    return (total_loss, logits, train_labels)


  def _forward(self, feed_dict):
    """
    One forward pass for the graph
    Args:
      feed_dict: feed_dict to the graph
    Return:
      a tuple of (loss_value, pred, labels)
      loss_value: average val loss
      pred: logits
      labels: labels of images
    """
    # Setup op_list
    op_list = [self.model.total_loss, \
               self.model.logits, \
               self.model.labels]

    # Forward pass
    loss_value, pred, labels = self.sess.run(op_list, feed_dict=feed_dict)
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    return (loss_value, pred, labels)


  def _add_image_summary(self, step, images, name, max_outputs=None):
    """Add image summary"""
    # TODO: finalize this function
    assert len(image.shape) == 4, \
        "Images must pack to (N, H, W, C) to be summarized"
    if max_outputs is None:
      max_outputs = self.batch_size
    im_pl = tf.placeholder(tf.float32, images.shape)
    summary_graph = tf.summary.image(name, im_pl, max_outputs)
    summary = self.sess.run(summary_graph, \
                            feed_dict={im_pl: images})
    self.summary_writer.add_summary(summary, step)

  
  def _add_graph_summary(self, step, summary_graph, feed_dict):
    """Add summary for the graph"""
    summary_str = self.sess.run(summary_graph, \
                                feed_dict=feed_dict)
    self.summary_writer.add_summary(summary_str, step)


  def _add_train_summary(self, step, train_feed_dict):
    """Add summary for train graph"""
    self._add_graph_summary(step, self.summary_op, \
                            train_feed_dict)
  

  def _add_eval_summary(self, step, loss, \
                        acc, iu, fps):
    """Add eval summary for graph"""
    feed_dict = {self.avg_loss_pl: loss, \
                 self.acc_total_pl: acc, \
                 self.mean_iu_pl: iu, \
                 self.fps_pl: fps}
    summary_op_list = [self.avg_loss_summary, \
                       self.acc_total_summary, \
                       self.mean_iu_summary, \
                       self.fps_summary]
    summary_str_list = self.sess.run(summary_op_list, \
                                     feed_dict=feed_dict)
    for summary_str in summary_str_list:
      self.summary_writer.add_summary(summary_str, step)


  def _save_onehot(self, preds, prefix):
    """Visualize and save label-like onehot-image"""
    # Visualize image preds / seq preds
    viz_func = lambda x: self._viz_tensor(x)
    viz_preds = map(viz_func, self._flatten(preds))
    viz_preds = np.stack(viz_preds)
    self._save_images(viz_preds, prefix)


  def _save_images(self, images, prefix, ext='png'):
    """Save images in png"""
    assert ext in ['jpg', 'png'], \
        "Invalid extension {}".format(ext)
    fname_format_str = prefix + '_{}.{}'
    for i,img in enumerate(images):
      img_name = fname_format_str.format(i, ext)
      img_path = os.path.join(self.predict_dir, \
                              img_name)
      cv2.imwrite(img_path, img)


  def _save_seqs(self, seqs, prefix):
    """Save image seqs as gif."""
    fname_format_str = prefix + '_{}.{}'
    for i,seq in enumerate(seqs):
      gif_name = fname_format_str.format(i, 'gif')
      gif_path = os.path.join(self.predict_dir, \
                              gif_name)
      images = np.split(seq, len(seq))
      imageio.mimsave(gif_path, images)


  def _viz_tensor(self, onehot_T):
    """Visualize an onehot-tensor into image"""
    valid_shape = (self.image_h, self.image_w, 1)
    assert onehot_T.shape == valid_shape, \
        "Tensor shape must be {}, but get {}".format(valid_shape, onehot_T.shape)
    onehot_T = onehot_T.squeeze()
    h, w = onehot_T.shape
    rgb = np.zeros((h, w, 3))
    for i,color in enumerate(self.cmap):
      rgb[onehot_T==i] = color
    return np.uint8(rgb)

  
  def _flatten(self, inputT):
    """Transform inputT into NHWC format"""
    assert len(inputT.shape) in (4, 5), \
        "Requires NHWC / NTHWC format"
    if len(inputT.shape) == 4:
      return inputT
    elif len(inputT.shape) == 5:
      N, T, H, W, C = inputT.shape
      new_shape = (N*T, H, W, C)
      return inputT.reshape(new_shape)
  

  def _print_train_info(self, step, loss_value, duration):
    format_str = ('{0}: step {1}, loss = {2:.2f} ({3:.1f}'
                  'examples/sec; {4:.3f} sec/batch)')
    fps = self.n_imgs / duration
    cost_time = float(duration)
    print format_str.format(datetime.now(), \
                            step, loss_value, \
                            fps, cost_time)


  def _train(self):
    """Run train graph"""
    cfg = self.cfg

    # Training loop
    start = self.startstep
    end = self.startstep + self.max_steps
    for step in range(start, end):
      # Optimization for one forward pass
      train_feed_dict = self._feed_dict(self.train_queue, \
                                             phase_train=True)
      start_time = time.time()
      loss_value, logits, train_labels = self._optimize(train_feed_dict)
      duration = time.time() - start_time

      # Print train info
      if step % 10 == 0:
        self._print_train_info(step, loss_value, duration)
        per_class_acc(self._flatten(logits), \
                      self._flatten(train_labels))

      # Validation & Write summary
      if step % 100 == 0:
        # Add summary for train graph
        self._add_train_summary(step, train_feed_dict)

        # Validating
        print "Start validating ..."
        self._eval(step=step, phase_val=True)
        print "end validating ..."
        
      # Save checkpoint
      if step % 1000 == 0 or (step + 1) == self.max_steps:
        self.saver.save(self.sess, self.checkpoint_path, global_step=step)


  def _eval(self, step=None, phase_val=False, \
            save_predict=False, verbose=True):
    """Testing the trained model"""
    cfg = self.cfg

    if phase_val:
      forward_iter = cfg.VAL_ITER
      input_queue = self.val_queue
    else:
      forward_iter = cfg.TEST_ITER
      input_queue = self.test_queue

    total_loss = 0.0
    duration = 0.0
    hist = np.zeros((self.n_classes, self.n_classes))
    all_preds = []
    all_labels = []
    for i in range(forward_iter):
      feed_dict = self._feed_dict(input_queue, \
                                  phase_train=False)
      start_time = time.time()
      loss, preds, labels = self._forward(feed_dict)
      duration += time.time() - start_time
      total_loss += loss
      if save_predict:
        arg_preds = np.argmax(preds, axis=-1)
        arg_preds = np.expand_dims(arg_preds, axis=-1)
        self._save_onehot(arg_preds, 'pred_{}'.format(i))
        self._save_onehot(labels, 'label_{}'.format(i))
      hist += get_hist(self._flatten(preds), \
                       self._flatten(labels))

    # Compute loss, acc, mean_iu, mean_fps
    avg_loss = total_loss / forward_iter 
    total_acc = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / \
         (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    mean_fps = self.n_imgs * forward_iter / duration

    # Add summary for test results
    if phase_val:
      assert step is not None, \
          "Must specify step in val phase"
      self._add_eval_summary(step, avg_loss, \
                             total_acc, mean_iu, \
                             mean_fps)

    # Print out info
    if verbose:
      if phase_val:
        print "val_loss: {}".format(avg_loss)
      print_hist_summery(hist)
    

  def _extract(self):
    raise NotImplementedError
