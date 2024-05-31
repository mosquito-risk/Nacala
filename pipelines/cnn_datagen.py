"""
Code for preparing dataset for CNN training.
The code can be used to create patches and labels for CNN models,
 generating DINOv2 feature vectors for training classic ML models.
"""
import glob
import torch
import os, sys
import rasterio
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import geometry_mask
from pipelines import dinv2_featuregen
from utils.rasterio_utils import new_row_column_offsets


class DataProcessingPipeline:
    def __init__(self, image_dir=None, label_dir=None, patch_size=448, stride=448, out_dir=None,
                 label_attribute_list: list = None, save_patches=False, create_dinov2_features=False,
                 dinov2_feature_file='temp.npy', convert_attr=None, label_format='shp'):
        self.label_df = None
        self.image_path = None
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.stride = stride
        self.out_dir = out_dir
        self.file_name_idx = 0
        self.save_patches = save_patches
        self.create_dinov2_features = create_dinov2_features
        self.label_attribute_list = label_attribute_list
        self.dinov2_features = None
        self.dinov2_feature_file = dinov2_feature_file
        self.convert_attr = convert_attr
        self.label_format = label_format

    def create_image_and_label_list(self):
        image_path_list = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        label_path_list = sorted(glob.glob(os.path.join(self.label_dir, '*.' + self.label_format)))
        print(f'Number of images : {len(image_path_list)} and Number of labels : {len(label_path_list)}')
        assert len(image_path_list) == len(label_path_list), f'Number of images and labels are not same'
        return image_path_list, label_path_list

    def image_transform(self, image_path):
        with rasterio.open(image_path) as src:
            return src.transform

    def get_projection_id(self):
        with rasterio.open(self.image_path) as src:
            return src.crs

    def create_window_per_polygon(self, image, label_df):
        params = []
        for _, row in label_df.iterrows():
            poly_geom = row.geometry
            wn = rasterio.windows.from_bounds(*poly_geom.bounds, transform=self.image_transform(image))
            col_off, row_off, patch_size = new_row_column_offsets(np.floor(wn.col_off), np.floor(wn.row_off),
                                                                  wn.height, wn.width)
            label_list = []
            if self.label_attribute_list:
                for attr in self.label_attribute_list:
                    label_list.append(row[attr])
            else:
                label_list = []
            params.append([poly_geom, col_off, row_off, patch_size, label_list])

        return params

    def create_batch(self, arg_list, image_path):
        """
        It takes list of window parameters and single batch of image patches and labels as a tensor
        """
        image_array = np.zeros((len(arg_list), 3, self.patch_size, self.patch_size))
        mask_array = np.zeros((len(arg_list), self.patch_size, self.patch_size))
        if self.label_attribute_list:
            label_array = np.zeros((len(arg_list), len(self.label_attribute_list)))
        else:
            label_array = None

        with rasterio.open(image_path) as src:
            for i, single_wn in tqdm(enumerate(arg_list)):
                poly_geom, col_off, row_off, patch_size, label_list = single_wn
                wn = Window(col_off, row_off, patch_size, patch_size)
                image_arr = src.read(window=wn, boundless=True, fill_value=0, indexes=[1, 2, 3])
                window_transform = rasterio.windows.transform(wn, src.transform)
                mask = geometry_mask([poly_geom], transform=window_transform, invert=True,
                                     out_shape=(patch_size, patch_size))

                # crtea numpy array of image and mask
                image_array[i] = image_arr
                mask_array[i] = mask
                if self.label_attribute_list:
                    label_array[i] = np.array(label_list)

        return image_array, mask_array, label_array

    def save_tensor(self, image_array, mask_array, label_array, out_dir):
        for i in range(len(image_array)):
            # save image, mask and label as pt file
            filename_ = f'{str(self.file_name_idx).zfill(8)}_{int(label_array[i][0])}.pt'
            file_path = os.path.join(out_dir, filename_)
            torch.save({
                'image': image_array[i],
                'mask': mask_array[i],
                'label': label_array[i]
            }, file_path)
            self.file_name_idx += 1

    def convert_attributes(self, label_df):
        # drop row with no label
        label_df = label_df.dropna(subset=self.convert_attr.keys())
        # convert attributes to int
        for attr in self.convert_attr:
            label_df[attr] = label_df[attr].replace(self.convert_attr[attr]['from'],
                                                    self.convert_attr[attr]['to'])
            label_df[attr] = label_df[attr].astype(float)
        # save label_df
        # label_df.to_file('../data/tanjania/raw_data/test/labels.geojson', driver='GeoJSON')

        return label_df

    def run_datagen_pipeline(self):
        image_path_list, label_path_list = self.create_image_and_label_list()
        print(f'Number of images : {len(image_path_list)} and Number of labels : {len(label_path_list)}')
        for image, label in zip(image_path_list, label_path_list):
            label_df = gpd.read_file(label)
            if label_df.shape[0] == 0:
                print(f'No label found in {label}')
                continue
            # check if projection is same
            # assert self.get_projection_id() == self.label_df.crs, f'Projection of image and label are not same'

            # check if attribute data list are int or string, if string then convert to int
            if self.convert_attr is not None:
                label_df = self.convert_attributes(label_df)

            # create window parameters
            arg_list = self.create_window_per_polygon(image, label_df)
            # create single batch from image
            image_array, mask_array, label_array = self.create_batch(arg_list, image)

            # save tensor
            if self.save_patches:
                self.save_tensor(image_array, mask_array, label_array, self.out_dir)

            # create dinov2 features
            if self.create_dinov2_features:
                features = dinv2_featuregen.DINOv2FeatureGen(image_array, mask_array, label_array).get_features()
                if self.dinov2_features is None:
                    self.dinov2_features = features
                else:
                    self.dinov2_features = np.vstack((self.dinov2_features, features))

        if self.create_dinov2_features:  # fixme write code to appeding data to file instead of to variable
            print(f'Saving DINOv2 features to {self.dinov2_feature_file}')
            print(f'Total features and points : {self.dinov2_features.shape}')
            np.save(self.dinov2_feature_file, self.dinov2_features)
