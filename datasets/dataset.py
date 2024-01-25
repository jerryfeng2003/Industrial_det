import torchvision.transforms
from PIL import Image
import os
import pickle as pkl
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomVerticalFlip, \
    RandomHorizontalFlip, RandomPerspective
from configs.config import project_root, data_root
from datasets.get_metadata import get_metadata
from timm.data import create_transform


# Deep Nutrient Deficiency dataset
class Dataset_image(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()

        if split not in ('trainval', 'train', 'val', 'test'):
            raise ValueError("split must be in trainval, train, val, or test. Supplied {}".format(split))
        self.split = split
        self.cfg = cfg

        self.img_paths, self.label_names, self.class_to_ind = get_metadata(data_root, split, verbose=True)
        self.label_idxs = [self.class_to_ind[i] for i in self.label_names] if self.label_names else None

        # use cache for acceleration
        self.cache_path = os.path.join(project_root, 'datasets/cache', str(self.cfg.im_scale))
        print("Cache path: ".ljust(20), self.cache_path)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # data augmentation
        if split.startswith('train'):

            """
             Create a RandAugment transform

             :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
             dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
             sections, not order sepecific determine
                 'm' - integer magnitude of rand augment
                 'n' - integer num layers (number of transform ops selected per image)
                 'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
                 'mstd' -  float std deviation of magnitude noise applied
                 'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
             Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
             'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2

             :param hparams: Other hparams (kwargs) for the RandAugmentation scheme

             :return: A PyTorch compatible Transform
             """
            # tform = create_transform(
            #     input_size=cfg.im_scale,
            #     is_training=True,
            #     color_jitter=0.4,
            #     auto_augment='rand-m1-mstd0.5-inc1',
            #     re_prob=0.25,
            #     re_mode='pixel',
            #     re_count=1,
            #     interpolation='bicubic',
            # )
            # self.tform_pipeline = tform
            tform = [
                torchvision.transforms.Resize(cfg.im_scale),
                RandomRotation(180),
                RandomHorizontalFlip(),
                # Cutout(),
                # RandomPerspective(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]

            self.tform_pipeline = Compose(tform)
        else:
            tform = [
                torchvision.transforms.Resize(cfg.im_scale),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]

            self.tform_pipeline = Compose(tform)

    def __getitem__(self, index):
        img_name = self.img_paths[index].split('/')[-1]
        assert os.path.isfile(self.img_paths[index]), self.img_paths[index] + " does not exit!"

        # use cache cuz loading and resizing the original image (high resolution) is time-consuming
        cache_file = os.path.join(self.cache_path, img_name[:-3] + 'pkl')
        if self.cfg.USE_CACHE and os.path.exists(cache_file):
            # if 0:
            with open(cache_file, 'rb') as f:
                img = pkl.load(f)
        else:
            img = Image.open(self.img_paths[index]).convert('RGB')

            # preprocessing (before data augmentation)
            img = img.resize((self.cfg.im_scale, self.cfg.im_scale), Image.BILINEAR)
            if self.cfg.USE_CACHE and not os.path.exists(cache_file):
                with open(cache_file, 'wb') as f:
                    pkl.dump(img, f)

        # preprocessing (data augmentation)
        if self.split != 'test':
            entry = {
                'img': self.tform_pipeline(img),
                'label_names': self.label_names[index] if self.label_names else None,
                'label_idxs': self.label_idxs[index] if self.label_idxs else None,
                'img_name': img_name,
                'img_path': self.img_paths[index]}
        else:
            entry = {
                'img': self.tform_pipeline(img),
                'img_name': img_name,
                'img_path': self.img_paths[index]}
        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.img_paths)


def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")
