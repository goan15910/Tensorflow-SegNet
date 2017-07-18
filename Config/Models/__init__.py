from simple_segnet_config import simple_segnet_config
from vgg16_lr_segnet_config import vgg16_lr_segnet_config
from vgg16_mr_segnet_config import vgg16_mr_segnet_config
from simple_lstm_segnet_config import simple_lstm_segnet_config

model_config_func_table = {
    'simple_segnet': simple_segnet_config, \
    'vgg16_segnet': simple_segnet_config, \
    'vgg16_lr_segnet': vgg16_lr_segnet_config, \
    'vgg16_mr_segnet': vgg16_mr_segnet_config, \
    'vgg16_partial_segnet': simple_segnet_config, \
    'simple_lstm_segnet': simple_lstm_segnet_config, \
}

def add_model_config(model_name, cfg):
  add_config_func = model_config_func_table[model_name]
  return add_config_func(cfg)
