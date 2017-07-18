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
from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io
from easydict import EasyDict as edict
# modules
from Utils import _variable_on_cpu, _variable_with_weight_decay, print_hist_summery, get_hist, per_class_acc, writeImage, generate_pretrained_initializer
from reader import Reader
from Config import set_config
from Nets import nets_table


def test(FLAGS):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir
  test_dir = FLAGS.test_dir
  test_ckpt = FLAGS.testing
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # testing should set BATCH_SIZE = 1
  batch_size = 1

  image_filenames, label_filenames = get_filename_list(test_dir)

  # Initialize model
  mc = model_config_table[FLAGS.net]
  model = nets_table[FLAGS.net](mc)

  # Inference
  logits = model.inference(test_data_node, test_labels_node, batch_size, phase_train)

  pred = tf.argmax(logits, dimension=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      mc.MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, test_ckpt )

    images, labels = get_all_test_data(image_filenames, label_filenames)

    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((mc.NUM_CLASSES, mc.NUM_CLASSES))
    for image_batch, label_batch  in zip(images, labels):

      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }

      dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
      # output_image to verify
      if (FLAGS.save_image):
          writeImage(im[0], 'testing_image.png')

      hist += get_hist(dense_prediction, label_batch)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))


def training(FLAGS, is_finetune=False):
  # Setup from cfg
  cfg = set_config(FLAGS.net, FLAGS.dataset)
  batch_size = cfg.BATCH_SIZE
  image_h = cfg.IMAGE_HEIGHT
  image_w = cfg.IMAGE_WIDTH
  image_c = cfg.IMAGE_DEPTH
  max_steps = cfg.MAX_STEPS

  # Setup from FLAGS
  train_dir = FLAGS.log_dir
  image_dir = FLAGS.image_dir
  val_dir = FLAGS.val_dir
  finetune_ckpt = FLAGS.finetune
  pretrained_path = FLAGS.pretrained
  # should be changed if your model stored by different convention
  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Setup reader
    reader = Reader(cfg, image_dir, 'train')
    if cfg.USE_LSTM:
      input_node = reader.seq_batch_node()
    else:
      input_node = reader.batch_node()

    # Setup pretrained initializer
    if pretrained_path is not None:
      pretrained_npy = np.load(pretrained_path)
      cfg.PRETRAINED_INITIALIZER = generate_pretrained_initializer(pretrained_npy)

    # Build the model graph
    model = nets_table[FLAGS.net](cfg)
    model.build()
    train_op = model.train(global_step)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
      # Build an initialization operation to run below.
      if (is_finetune == True):
        saver.restore(sess, finetune_ckpt)
      else:
        init = tf.global_variables_initializer()
        sess.run(init)

      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      # Summery placeholders
      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.summary.scalar("test_average_loss", average_pl)
      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

      for step in range(startstep, startstep + max_steps):
        # Get feed batch for the graph
        image_batch ,label_batch = sess.run(input_node)
        feed_dict = {
          model.images_node: image_batch,
          model.labels_node: label_batch,
          model.phase_train: True
        }

        start_time = time.time()

        # Forward pass and optimize
        loss_value, _, train_labels = sess.run([model.total_loss, 
                                                 train_op, 
                                                 model.labels], 
                                               feed_dict=feed_dict)

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Print stdout information
        if step % 10 == 0:
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch per-class accuracy
          pred = sess.run(model.logits, feed_dict=feed_dict)
          per_class_acc(pred, train_labels)

        if step % 100 == 0:
          print("start validating.....")
          total_val_loss = 0.0
          hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
          for test_step in range(cfg.TEST_ITER):
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
            val_feed_dict = {
                model.images_node: val_images_batch,
                model.labels_node: val_labels_batch,
                model.phase_train: True
            }
            _val_loss, _val_pred, _val_labels = sess.run([model.total_loss,
                                                           model.logits,
                                                           model.labels],
                                                         feed_dict=val_feed_dict)
            total_val_loss += _val_loss
            hist += get_hist(_val_pred, _val_labels)
          print("val loss: ", total_val_loss / cfg.TEST_ITER)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / cfg.TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          print_hist_summery(hist)
          print(" end validating.... ")

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
          checkpoint_path = os.path.join(train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

      coord.request_stop()
      coord.join(threads)
