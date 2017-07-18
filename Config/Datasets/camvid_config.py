from easydict import EasyDict as edict
import numpy as np 

def camvid_config(cfg):

  # Customize dc here
  # Example: dc.xxx = xxx
  # Dataset dependent information
  cfg.DATASET_NAME = "CamVid"
  cfg.IMAGE_HEIGHT = 360
  cfg.IMAGE_WIDTH = 480
  cfg.IMAGE_DEPTH = 3
  cfg.NUM_CLASSES = 11
  cfg.NUM_TRAIN_EXAMPLES_PER_EPOCH = 367
  cfg.NUM_TEST_EXAMPLES_PER_EPOCH = 101
  cfg.NUM_VAL_EXAMPLES_PER_EPOCH = 1
  cfg.TEST_ITER = cfg.SET_ITER_FUNC(cfg.NUM_TEST_EXAMPLES_PER_EPOCH, \
                                    cfg.BATCH_SIZE)
  cfg.VAL_ITER = cfg.SET_ITER_FUNC(cfg.NUM_VAL_EXAMPLES_PER_EPOCH, \
                                   cfg.BATCH_SIZE)
  cfg.LOSS_WEIGHT = np.array([
       0.2595, 0.1826, 4.5640,
       0.1417, 0.9051, 0.3826,
       9.6446, 1.8418, 0.6823,
       6.2478, 7.3614,]) # class 0~10
  cfg.PARSE_LINE_FUNCTION = parse_line

  return cfg


def parse_line(line):
  """Pasrse the line of format of CamVid dataset"""
  img_path, label_path = line.strip().split(' ')
  img_fname = img_path.split('/')[-1]
  clip_name = img_fname.split('_')[0]
  return (clip_name, img_path, label_path)
