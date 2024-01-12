from easydict import EasyDict
from pathlib import Path
import numpy as np
import os
import yaml
import argparse

project_root = Path(__file__).resolve().parents[1]

data_root = './data'

cfg = EasyDict()

# init
cfg.is_test = False
cfg.use_cache = False
cfg.p_interval = 100
cfg.ngups = 1
cfg.im_scale = 256
cfg.bsz = 4
cfg.num_classes = 5
cfg.acc_bsz = 1
cfg.train_set = 'train'
cfg.model_name = 'swin_v2_b'

# models:
# Convnet:
# resnet,efficientnet ,  convnext, convnext v2,
# Transformer:
# vit(scales), swin, maxvit, xxvit
# Convnet + Transformer:
# Coatnet, Coatnetv2, .....
# Self supervised model:
# beit, mae ....

# Optimize ,lr strategies:....

# lr adjust: EMA,..

# transfer learning trick: ML-Encoder ....

# Data argumentations: ...

# Model ensemble: vote, add, (ME tactics...)

# read papers !!!

cfg.from_scratch = True
cfg.model_path = cfg.model_name + '.pth'
if cfg.from_scratch and cfg.is_test == False:
    cfg.restore_from = './pretrain/' + cfg.model_path
else:
    cfg.restore_from = './checkpoints/best_model_' + cfg.model_path

# COMMON CONFIGS
cfg.SEED = 114514  # 1234, 3407, 42
cfg.NWORK = 4
cfg.USE_CACHE = cfg.use_cache

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.OPTIM = 'adam'  # sgd | adam | rmsprop
# Network params
cfg.TRAIN.LR = 1e-4  # 1e-4 1e-3
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 1e-5  # 0.0005， L2
cfg.TRAIN.POWER = 0.9
# cfg.TRAIN.CLIP = 5.0          # gradients will be clipped to have norm less than this
# Other params
cfg.TRAIN.MAX_EPOCH = 100

# TEST CONFIGS
cfg.TEST = EasyDict()


# Network params
# cfg.TEST.MODEL_WEIGHT = (1.0,)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)

    return cfg


def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def yaml_dump(python_object, file_path):
    make_parent(file_path)
    with open(file_path, 'w') as f:
        yaml.dump(python_object, f, default_flow_style=False)


def make_parent(file_path):
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)
