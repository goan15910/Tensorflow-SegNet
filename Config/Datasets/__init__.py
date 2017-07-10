from camvid_config import camvid_config

dataset_config_func_table = {
    'camvid': camvid_config, \
}

def add_dataset_config(dataset_name, cfg):
  add_config_func = dataset_config_func_table[dataset_name]
  return add_config_func(cfg)
