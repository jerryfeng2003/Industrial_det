import os
from collections import Counter
from configs.config import data_root
import yaml

class_to_ind = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
}


def get_metadata(data_root, split='train', verbose=True):
    data_path = os.path.join(data_root, 'images')
    print("Data root path: ".ljust(40), data_path)
    assert os.path.exists(data_path), "{} does not exist, please check your root_path".format(data_path)

    # load split
    split_path = os.path.join(data_root, split) + '.txt'
    with open(split_path, 'r') as f:
        file_names = f.read().splitlines()
        print("Loading split from: ".ljust(40), split_path)

    # load metadata
    labels = None
    if split != 'test':
        metadata_path = os.path.join(data_root, 'labels_trainval.yaml')
        labels_trainval = yaml.safe_load(open(metadata_path, 'r'))
        print("Loading labels from: ".ljust(40), metadata_path)

        labels = [labels_trainval[file_name] for file_name in file_names]

    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    if verbose:
        print("Num of images: ".ljust(40), len(file_paths))
        print("Num of labels: ".ljust(40), sum(Counter(labels).values()), len(Counter(labels).values()),
              Counter(labels))

    return file_paths, labels, class_to_ind


if __name__ == '__main__':
    split = 'train'  # train,val, test
    img_paths, label_names, class_to_ind = get_metadata(data_root, split, verbose=True)
