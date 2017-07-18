from easydict import EasyDict as edict

def vgg16_lr_segnet_config(cfg):

  # Customize mc here
  # Example: mc.xxx = xxx
  cfg.DOWN_SAMPLE_RATIO = 4

  return cfg
