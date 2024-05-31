import os
import shutil


def create_folder(dir_):
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)


def create_multi_folder(dir_, exists_ok=True):
    os.makedirs(dir_, exist_ok=exists_ok)


def custom_tree_folder(dir_, folder_type='train', weights=False):
    if folder_type == 'train':
        create_multi_folder(os.path.join(*[dir_, 'train', 'images']))
        create_multi_folder(os.path.join(*[dir_, 'train', 'labels']))
        create_multi_folder(os.path.join(*[dir_, 'train', 'p_labels']))
        if weights:
            create_multi_folder(os.path.join(*[dir_, 'train', 'weights']))
            create_multi_folder(os.path.join(*[dir_, 'train', 'p2_labels']))
    elif folder_type == 'valid':
        create_multi_folder(os.path.join(*[dir_, 'valid', 'images']))
        create_multi_folder(os.path.join(*[dir_, 'valid', 'labels']))
        create_multi_folder(os.path.join(*[dir_, 'valid', 'p_labels']))
        if weights:
            create_multi_folder(os.path.join(*[dir_, 'valid', 'weights']))
            create_multi_folder(os.path.join(*[dir_, 'valid', 'p2_labels']))
    elif folder_type == 'test':
        create_multi_folder(os.path.join(*[dir_, 'test', 'images']))
        create_multi_folder(os.path.join(*[dir_, 'test', 'labels']))
        create_multi_folder(os.path.join(*[dir_, 'test', 'p_labels']))
        if weights:
            create_multi_folder(os.path.join(*[dir_, 'test', 'weights']))
            create_multi_folder(os.path.join(*[dir_, 'test', 'p2_labels']))
