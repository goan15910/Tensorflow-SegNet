from Models import add_model_config
from Datasets import add_dataset_config
from base_config import cfg, set_iteration

def set_config(model_name, dataset_name):
  cfg = add_model_config(model_name, cfg)
  cfg = add_dataset_config(dataset_name, cfg)
  return cfg
