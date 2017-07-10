from easydict import EasyDict as edict

def simple_lstm_segnet_config(cfg):

  # Customize mc here
  # Example: mc.xxx = xxx
  cfg.N_STATE = 64

  return cfg
