from Models import add_model_config
from Datasets import add_dataset_config
from base_config import cfg

def set_config(model_name, dataset_name):
  mc = add_model_config(model_name, cfg)
  dmc = add_dataset_config(dataset_name, mc)
  return dmc
