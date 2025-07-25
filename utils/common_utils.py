import os
import torch
import shutil


def apply_color_map(predictions):
    # Initialize an empty image with the shape [batch_size, 3, height, width]
    batch_size, height, width = predictions.shape
    colored_images = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8,
                                 device=predictions.device)
    # Define your color map: each class is assigned a unique color
    color_map = {
        0: [255, 0, 0],  # Red
        1: [0, 255, 0],  # Green
        2: [0, 0, 255],  # Blue
        3: [255, 255, 0],  # Yellow
        4: [255, 0, 255],  # Magenta
        5: [0, 255, 255]  # Cyan
    }
    for cls, color in color_map.items():
        if cls == 0:
            continue
        mask = (predictions == cls).unsqueeze(1)  # Add channel dimension
        for i in range(3):  # For each color channel
            colored_images[:, i, :, :] += mask[:, 0, :, :] * color[i]

    return colored_images


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
