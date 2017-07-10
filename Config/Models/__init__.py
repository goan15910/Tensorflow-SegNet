from simple_segnet_config import simple_segnet_config
from vgg16_segnet_config import vgg16_segnet_config
from vgg16_lr_segnet_config import vgg16_lr_segnet_config
from vgg16_mr_segnet_config import vgg16_mr_segnet_config

model_config_func_table = {
    'simple_segnet': simple_segnet_config, \
    'vgg16_segnet': vgg16_segnet_config, \
    'vgg16_lr_segnet': vgg16_lr_segnet_config, \
    'vgg16_mr_segnet': vgg16_mr_segnet_config, \
    'vgg16_partial_segnet': vgg16_segnet_config, \
}

def add_model_config(model_name, cfg):
  add_config_func = model_config_func_table[model_name]
  return add_config_func(cfg)
