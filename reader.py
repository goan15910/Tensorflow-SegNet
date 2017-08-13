import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import skimage
import skimage.io


class Reader:
  """Base class of dataset reader"""
  def __init__(self, cfg, image_dir, phase):
    # Setup image directory
    self.image_dir = image_dir

    # Setup phase
    self.phase = phase

    # Model params
    self.batch_size = cfg.BATCH_SIZE
    self.use_lstm = cfg.USE_LSTM
    self.seq_len = cfg.SEQUENCE_LENGTH

    # Dataset params
    self.dataset_name = cfg.DATASET_NAME
    self.n_classes = cfg.NUM_CLASSES
    self.im_h = cfg.IMAGE_HEIGHT
    self.im_w = cfg.IMAGE_WIDTH
    self.im_c = cfg.IMAGE_DEPTH
    self.im_shape = (self.im_h, self.im_w, self.im_c)
    self.la_shape = (self.im_h, self.im_w, 1)
    if self.phase == 'train':
      self.n_examples_per_epoch = cfg.NUM_TRAIN_EXAMPLES_PER_EPOCH
    elif self.phase == 'test':
      self.n_examples_per_epoch = cfg.NUM_TEST_EXAMPLES_PER_EPOCH
    elif self.phase == 'val':
      self.n_examples_per_epoch = cfg.NUM_VAL_EXAMPLES_PER_EPOCH
    self._parse_line_func = cfg.PARSE_LINE_FUNCTION

    # Queue dependent info
    self.min_queue_fraction = cfg.MIN_QUEUE_FRACTION
    self.min_queue_examples = int(self.n_examples_per_epoch * \
                                  self.min_queue_fraction)

    # Print info
    self.enqueue_info_str = "Filling queue with {} {} images before starting to train. This will take a few minutes."
    

  def batch_node(self):
    """Set the batch input node for graph"""
    # Set filename list
    image_filenames, label_filenames = self._get_fnames(self.image_dir)
    n_instances = len(image_filenames)

    # Read in images for generating batch
    scope_name = 'batch_input_{}'.format(self.phase)
    with tf.variable_scope(scope_name) as scope:
      images, labels = [], []
      for i in xrange(n_instances):
        image, label = self._read_image([image_filenames[i], label_filenames[i]])
        image = tf.cast(image, tf.float32)
        images.append(image)
        labels.append(label)

      print self.enqueue_info_str.format(self.min_queue_examples, \
                                         self.dataset_name)

      # Generate a batch of images and labels by building up a queue of examples.
      return self._generate_batch(images, labels, shuffle=True)


  def seq_batch_node(self):
    """Set the sequence batch input node for graph"""
    # Set filename list for filename queue
    image_fname_seqs, label_fname_seqs = self._get_seq_fnames(self.image_dir)
    n_seqs = len(image_fname_seqs)

    # Read in sequences for generating batch
    scope_name = 'seq_batch_input_{}'.format(self.phase)
    with tf.variable_scope(scope_name) as scope:
      image_seqs, label_seqs = [], []
      for i in xrange(n_seqs):
        image_seq, label_seq = self._read_seq(image_fname_seqs[i], \
                                              label_fname_seqs[i])
        image_seq = tf.cast(image_seq, tf.float32)
        image_seqs.append(image_seq)
        label_seqs.append(label_seq)

      image_seqs = tf.stack(image_seqs)
      label_seqs = tf.stack(label_seqs)

      print self.enqueue_info_str.format(self.min_queue_examples, \
                                         self.dataset_name)

      # Generate a batch of images and labels by building up a queue of examples.
      return self._generate_batch(image_seqs, label_seqs, shuffle=True)


  def _read_image(self, filename_pair):
    """Read the one image and its label"""
    image_filename = filename_pair[0]
    label_filename = filename_pair[1]

    imageValue = tf.read_file(image_filename)
    labelValue = tf.read_file(label_filename)

    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)

    image = tf.reshape(image_bytes, self.im_shape)
    label = tf.reshape(label_bytes, (self.im_h, self.im_w, 1))

    return image, label


  def _read_seq(self, image_fname_seq, label_fname_seq, padding=True):
    """Read the one image seq and its label seq"""
    n_frame = len(image_fname_seq)
    pad_len = self.seq_len - n_frame

    image_seq, label_seq = [], []
    for i in xrange(n_frame):
      fname_pair = (image_fname_seq[i], \
                    label_fname_seq[i])
      image, label = self._read_image(fname_pair)
      image_seq.append(image)
      label_seq.append(label)

    if pad_len != 0:
      pad_frame = image_seq[-1]
      pad_label = label_seq[-1]
      pad_frames = [pad_frame] * pad_len
      pad_labels = [pad_label] * pad_len
      image_seq.extend(pad_frames)
      label_seq.extend(pad_labels)
    
    image_seq = tf.stack(image_seq)
    label_seq = tf.stack(label_seq)

    return image_seq, label_seq


  def _generate_batch(self, imageT, labelT, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      images: Image tensors of [N, ...] of type.float32.
      labels: Label tensors of [N, ...] type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Batch size.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      imageT_batch: Image tensors of [batch_size, ...] size.
      labelT_batch: Label tensors of [batch_size, ...] size.
    """
    # Create a queue that shuffles the examples,
    # and then dequeue a batch of images and labels
    min_queue_examples = self.min_queue_examples
    batch_size = self.batch_size
    num_preprocess_threads = 1
    capacity = min_queue_examples + 3 * batch_size

    with tf.variable_scope('batch_queue') as scope:
      if shuffle:
        imageT_batch, labelT_batch = tf.train.shuffle_batch(
            [imageT, labelT],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            enqueue_many=True)
      else:
        imageT_batch, labelT_batch = tf.train.batch(
            [imageT, labelT],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            enqueue_many=True)
    
    return imageT_batch, labelT_batch


  def _get_fnames(self, path):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
      i = i.strip().split(" ")
      image_filenames.append(i[0])
      label_filenames.append(i[1])
    return image_filenames, label_filenames


  def _get_seq_fnames(self, path):
    with open(path) as fd:
      lines = fd.readlines()

    image_seqs, label_seqs = [], [] # list of image and label seq
    im_seq = [] # image fnames tmp list
    la_seq = [] # label fnames tmp list
    last_clip = None
    clip_len = 0
    for line in lines:
      cur_clip, im_path, la_path = self._parse_line_func(line)
      if (cur_clip != last_clip) or (clip_len == self.seq_len):
        if last_clip is not None:
          image_seqs.append(im_seq)
          label_seqs.append(la_seq)
        im_seq = [im_path]
        la_seq = [la_path]
        clip_len = 1
      elif (cur_clip == last_clip):
        im_seq.append(im_path)
        la_seq.append(la_path)
        clip_len += 1
      last_clip = cur_clip
    image_seqs.append(im_seq)
    label_seqs.append(la_seq)

    return (image_seqs, label_seqs)


  def _get_all_test_data(self, im_list, la_list):
    images = []
    labels = []
    index = 0
    for im_filename, la_filename in zip(im_list, la_list):
      im = np.array(skimage.io.imread(im_filename), np.float32)
      im = im[np.newaxis]
      la = skimage.io.imread(la_filename)
      la = la[np.newaxis]
      la = la[...,np.newaxis]
      images.append(im)
      labels.append(la)
    return images, labels
