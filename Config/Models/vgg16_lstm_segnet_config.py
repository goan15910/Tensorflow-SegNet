from easydict import EasyDict as edict

def vgg16_lstm_segnet_config(cfg):

  # Customize mc here
  # Example: mc.xxx = xxx
  cfg.BATCH_SIZE = 2
  cfg.USE_LSTM = True
  cfg.SEQUENCE_LENGTH = 3

  return cfg
