import os
import glob
import torch


class BaseClass:
    def __init__(self, image_dir=None, label_dir=None, **kwargs):
        self.label_dir = label_dir
        self.image_dir = image_dir
        if self.label_dir is None:
            self.label_dir = self.image_dir
    def create_image_list(self):
        image_path_list = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        print(f'Number of images : {len(image_path_list)}')
        assert len(image_path_list) != 0, f'There are no image in the directory, please check the path'
        return image_path_list

    def create_label_list(self):
        label_path_list = sorted(glob.glob(os.path.join(self.label_dir, '*.geojson')))
        print(f'Number of labels : {len(label_path_list)}')
        assert len(label_path_list) != 0, f'Number of images and labels are not same'
        return label_path_list

    def get_device(self):
        device = "cpu"
        if torch.cuda.is_available():
            print("CUDA is available..")
            device = "cuda"
        return device

