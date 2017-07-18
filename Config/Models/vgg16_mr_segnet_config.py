from easydict import EasyDict as edict

def vgg16_mr_segnet_config(cfg):

  # Customize mc here
  # Example: mc.xxx = xxx
  cfg.DOWN_SAMPLE_RATIO = 2

  return cfg
