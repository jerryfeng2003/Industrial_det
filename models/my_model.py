import torch
import torch.nn as nn
import torchvision.models as models
import timm

import configs.config
from utils.pytorch_misc import *


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.num_classes = cfg.num_classes
        self.model_name = cfg.model_name

        print("Loading pretrained: ", self.model_name)
        self.model = getattr(models, self.model_name)(weights='DEFAULT')
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

    def forward(self, x):
        return self.model(x)
