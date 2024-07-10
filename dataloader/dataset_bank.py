from os import listdir
from os.path import join
from ipdb import set_trace as st
import glob
import os
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset_std(root, data_split, tasks):
    input_list = sorted(glob.glob(join(root, 'rgb', data_split, '*.JPG')))
    aif_list = sorted(glob.glob(join(root, 'aif', data_split, '*.jpg')))
    targets_list = []
    for task in tasks:
        targets_list.append(sorted(glob.glob(join(root, task, data_split, '*.png'))))
    # return list(zip(input_list, targets_list))
    return input_list, targets_list, aif_list

def dataset_target_only(root, phase):
    return sorted(glob.glob(join(root, 'depth', phase, '*.png')))


def dataset_simcol3d(root, data_split, tasks):
    print(os.path.exists(join(root, 'rgb', data_split)))
    input_list = sorted(glob.glob(join(root, 'rgb_30mm', data_split, '*.png')))
    targets_list = sorted(glob.glob(join(root, 'rgb_20mm', data_split, '*.png')))
    depths_list = sorted(glob.glob(join(root, 'depth', data_split, '*.png')))
    aif_list = sorted(glob.glob(join(root, 'aif', data_split, '*.png')))

    # return list(zip(input_list, targets_list))
    return input_list, targets_list, depths_list, aif_list
