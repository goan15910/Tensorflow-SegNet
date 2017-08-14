from easydict import EasyDict as edict

def simple_lstm_segnet_config(cfg):

  # Customize mc here
  # Example: mc.xxx = xxx
  cfg.USE_LSTM = True
  cfg.SEQUENCE_LENGTH = 3

  return cfg
