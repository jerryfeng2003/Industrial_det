import torch
import torch.nn as nn
import torchvision.models as models
from models.Point_MAE import PointTransformer
from models import models_vit
from utils.pos_embed import interpolate_pos_embed
import timm

import configs.config
from utils.pytorch_misc import *


class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.model_name = cfg.model_name

        print("Loading pretrained: ", self.model_name)
        if self.model_name == 'swin_v2_b':
            self.model = getattr(models, self.model_name)(weights='DEFAULT')
            self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
        elif self.model_name == 'mae':
            self.model = models_vit.__dict__['vit_base_patch16']()
            checkpoint = torch.load(cfg.restore_from, map_location='cuda')
            checkpoint_model = checkpoint['model']
            interpolate_pos_embed(self.model, checkpoint_model)
            self.model.load_state_dict(checkpoint_model, strict=False)
            self.model.head = nn.Linear(self.model.embed_dim, self.num_classes)

    def forward(self, x):
        return self.model(x)


class MyModel_point(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.model_name = cfg.model_name

        print("Loading pretrained: ", self.model_name)
        self.model = PointTransformer(num_class=self.num_classes)

    def forward(self, x):
        return self.model(x)
