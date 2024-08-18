"""
Data preparation for different models
Code for splitting big image to small patches and create different label formats such as patch labels, coco and yolo.
Currently implemented for only coco, patch and yolo will be added soon.
"""

import os
import glob

import numpy as np
import rasterio
import pandas as pd
from itertools import product
from rasterio.windows import Window
import geopandas as gpd
from tqdm import tqdm
from rasterio.io import MemoryFile

from utils.vector_utils import clip_polygon
from utils.common_utils import create_folder
from utils.unet_weight_mask import (border_weight_map_rio, interior_polygons, energy_levels_with_distance,
                                    interior_polygons_with_euc)
from utils.yolo_format_utils import convert_coco_json, create_yml_file, get_class_names
from utils.rasterio_utils import window_to_polygon, get_transform, rasterize_gdf_within_window
from utils.coco_format_utils import coco_annotation_dict, custom_coco_from_polygons, save_coco_json



class patch_gen:
    def __init__(self, image_dir: str, label_dir: str, out_folder: str, image_format: str = 'tif',
                 label_format: str = 'shp', patch_size: int = 2048, overlap: int = 0, coco_labels: bool = False,
                 yolo_labels: bool = False, yolo_binary=False, patch_labels: bool = False, label_attribute: str = None,
                 band_list=None, weight_mask=True, convert_attr=None, geojson_labels=False, data_per_thresh=0.8,
                 coco_category_dict='nacala', int_mask=False, energy_mask=False, level_dist=4,
                 w0=10, sigma=5, write_images=True, inter_per=70, int_mask_euc=False, exterior_dist=5,
                 subset_info=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_format = image_format
        self.label_format = label_format
        self.patch_size = patch_size
        self.stride = self.patch_size - overlap
        self.label_attribute = label_attribute
        self.start_ann_id = 1
        self.yolo_binary = yolo_binary
        self.start_index = 0
        self.convert_attr = convert_attr
        self.coco_category_dict = coco_category_dict
        self.patch_rows = []
        self.patch_geo = []
        self.w0 = w0
        self.data_per_thresh = data_per_thresh
        self.sigma = sigma

        if band_list is None:
            self.band_list = (1, 2, 3)
        else:
            self.band_list = band_list
        self.out_folder = out_folder
        assert os.path.exists(self.out_folder), f'{self.out_folder} does not exist'
        self.write_images = write_images
        if self.write_images:
            create_folder(os.path.join(self.out_folder, 'images'))
        self.patch_labels = patch_labels
        if self.patch_labels:
            create_folder(os.path.join(self.out_folder, 'p_labels'))
        self.geojson_labels = geojson_labels
        if self.geojson_labels:
            create_folder(os.path.join(self.out_folder, 'geojson'))
        self.int_mask = int_mask
        if self.int_mask:
            self.inter_folder_name = f'int_mask{inter_per}'
            create_folder(os.path.join(self.out_folder, self.inter_folder_name))
            self.inter_per = inter_per / 100
        self.int_mask_euc = int_mask_euc
        if self.int_mask_euc:
            self.int_euc_folder_name = f'int_mask_euc{exterior_dist}'
            create_folder(os.path.join(self.out_folder, self.int_euc_folder_name))
            self.exterior_dist = exterior_dist
        self.weight_mask = weight_mask
        if self.weight_mask:
            create_folder(os.path.join(self.out_folder, 'weights'))
        self.energy_mask = energy_mask
        if self.energy_mask:
            self.level_dist = level_dist
            create_folder(os.path.join(self.out_folder, f'energy_mask{self.level_dist}'))
        self.patch_boundaries = os.path.join(self.out_folder, 'patch_boundaries.shp')
        self.image_data_boundary = os.path.join(self.out_folder, 'image_data_boundary.shp')
        self.yolo_labels = yolo_labels
        self.coco_labels = coco_labels
        if self.coco_labels or self.yolo_labels or self.yolo_binary:
            self.coco_annotations_file = os.path.join(self.out_folder, 'annotation.json')
            # create empty lists for image and annotation dictionaries (for coco format)
            self.final_image_dicts = []
            self.final_annotation_dicts = []
            if self.yolo_labels:
                self.yolo_label_folder = os.path.join(self.out_folder, 'labels')
                create_folder(self.yolo_label_folder)
            if self.yolo_binary:
                self.yolo_binary_folder = os.path.join(self.out_folder, 'labels_binary')
                create_folder(self.yolo_binary_folder)
        if subset_info is not None:
            self.subset_info = subset_info
            _gdf = gpd.read_file(subset_info)
            _gdf = _gdf[_gdf["scale_law"] == 1]
            self.subset_files = _gdf["train_file"].tolist()
            print(f'Number of subset files: {len(self.subset_files)}')

    def convert_attributes(self, label_df):
        # drop row with no label
        label_df = label_df.dropna(subset=self.convert_attr.keys())
        # convert attributes to int
        for attr in self.convert_attr:
            label_df[attr] = label_df[attr].replace(self.convert_attr[attr]['from'],
                                                    self.convert_attr[attr]['to'])
            label_df[attr] = label_df[attr].astype(float)
        return label_df

    def patch_gen(self, image_path: str, label_path: str = None):
        # get offset and size of the image
        img = rasterio.open(image_path, tiled=True, blockxsize=256, blockysize=256)
        img_height, img_width = img.height, img.width
        patch_offsets = list(product(range(0, img_width, self.stride), range(0, img_height, self.stride)))
        big_window = Window(0, 0, img_width, img_height)
        # image_boundary(image_path, self.image_data_boundary)
        # im_boundary = gpd.read_file(self.image_data_boundary)
        label_df = gpd.read_file(label_path)
        if self.convert_attr is not None:
            label_df = self.convert_attributes(label_df)

        img_transform = get_transform(image_path)
        for i, (col_off, row_off) in enumerate(patch_offsets):
            patch_window = Window(col_off=col_off, row_off=row_off, width=self.patch_size,
                                  height=self.patch_size)
            patch = img.read(window=patch_window, boundless=True, fill_value=0, indexes=list(self.band_list))
            patch_polygon = window_to_polygon(img_transform, patch_window)
            patch_gdf = clip_polygon(patch_polygon, label_df)
            patch_transform = rasterio.windows.transform(patch_window, img_transform)
            if patch_gdf.shape[0] > 0:
                patch_label = rasterize_gdf_within_window(patch_window, patch_transform, patch_gdf,
                                                          self.label_attribute)
            else:
                patch_label = np.zeros((self.patch_size, self.patch_size))

            # save bbox of patch and bbox attributes
            data_mask = patch[0, :, :] != 0
            data_per = data_mask.sum() / (self.patch_size * self.patch_size)
            if data_per != 0:
                valid_patch = 1
                filename_ = f'{str(self.start_index).zfill(8)}.tif'
                if data_per >= self.data_per_thresh:
                    valid_patch = 2

                    if self.write_images:
                        filename = os.path.join(*[self.out_folder, 'images', filename_])
                        with rasterio.open(filename, 'w', driver='GTiff', height=self.patch_size,
                                           width=self.patch_size, count=3, dtype=patch.dtype,
                                           crs=img.crs, transform=patch_transform) as dst:
                            dst.write(patch)

                    # write raster patch label within the image window
                    if self.patch_labels:
                        filename_l = os.path.join(*[self.out_folder, 'p_labels', filename_])
                        with rasterio.open(filename_l, 'w', driver='GTiff', height=self.patch_size,
                                           width=self.patch_size, count=1, dtype=patch_label.dtype,
                                           crs=img.crs, transform=patch_transform) as dst:
                            dst.write(np.expand_dims(patch_label, axis=0))

                    # write geojson patch label
                    if self.geojson_labels:
                        filename_g = os.path.join(*[self.out_folder, 'geojson', filename_.replace('tif', 'geojson')])
                        patch_gdf.to_file(filename_g, driver="GeoJSON")  # patch geojson
                    # write weight mask and updated patch labels
                    if self.weight_mask:
                        if data_per < 1:
                            weights = data_mask.astype(np.uint8)
                        else:
                            weights = np.ones((self.patch_size, self.patch_size)).astype(np.uint8)
                        if patch_gdf.shape[0] > 0:
                            unet_weights = border_weight_map_rio(patch_gdf,
                                                                 height=self.patch_size,
                                                                 width=self.patch_size,
                                                                 transform=patch_transform,
                                                                 w0=self.w0,
                                                                 sigma=self.sigma)
                            weights *= unet_weights

                        filename_m = os.path.join(*[self.out_folder, 'weights', filename_])
                        with rasterio.open(filename_m, 'w', driver='GTiff', height=self.patch_size,
                                           width=self.patch_size, count=1, dtype=weights.dtype,
                                           crs=img.crs, transform=patch_transform) as dst:
                            dst.write(np.expand_dims(weights, axis=0))

                    # write coco and yolo format labels
                    if self.coco_labels or self.yolo_labels or self.yolo_binary:  # patch coco labels
                        image_dict = {"file_name": filename_, "height": self.patch_size, "width": self.patch_size,
                                      "id": int(self.start_index)}
                        self.final_image_dicts.append(image_dict)
                        # rasterio transform to gdal transform
                        gdal_transform = (patch_transform.c, patch_transform.a, patch_transform.b, patch_transform.f,
                                          patch_transform.d, patch_transform.e)
                        if patch_gdf.shape[0] > 0:
                            ann_dict, start_ann_id = coco_annotation_dict(patch_gdf, self.start_index, gdal_transform,
                                                                          self.label_attribute, self.start_ann_id)
                            self.final_annotation_dicts += ann_dict
                    # interior and exterior labels
                    if self.int_mask:
                        patch_window_ = Window(col_off=col_off, row_off=row_off,
                                               width=self.patch_size,
                                               height=self.patch_size)
                        patch_ = img.read(window=patch_window_, boundless=True, fill_value=0,
                                          indexes=list(self.band_list))
                        _, row_, col_ = patch_.shape
                        patch_polygon_ = window_to_polygon(img_transform, patch_window_)
                        # get touching polygons with patch polygon

                        touched_gdf = label_df[label_df.geometry.intersects(patch_polygon_)]
                        patch_transform_ = rasterio.windows.transform(patch_window_, img_transform)
                        if touched_gdf.shape[0] > 0:
                            interior_gdf = interior_polygons(touched_gdf, self.inter_per, self.label_attribute)
                            patch_gdf_ = clip_polygon(patch_polygon_, interior_gdf)
                            if patch_gdf_.shape[0] > 0:
                                int_mask_ = rasterize_gdf_within_window(patch_window_, patch_transform_, patch_gdf_,
                                                                        self.label_attribute)
                            else:
                                int_mask_ = np.zeros((self.patch_size, self.patch_size))
                        else:
                            int_mask_ = np.zeros((self.patch_size, self.patch_size))
                        # save interior and exterior labels
                        filename_int = os.path.join(*[self.out_folder, self.inter_folder_name, filename_])
                        with rasterio.open(filename_int, 'w', driver='GTiff', height=self.patch_size,
                                           width=self.patch_size, count=1, dtype=rasterio.uint8,
                                           crs=img.crs, transform=patch_transform) as dst:
                            dst.write(np.expand_dims(int_mask_, axis=0))
                    # interior mask based on the euclidian distance
                    if self.int_mask_euc:
                        patch_window2 = Window(col_off=col_off, row_off=row_off,
                                               width=self.patch_size,
                                               height=self.patch_size)
                        patch_polygon_ = window_to_polygon(img_transform, patch_window2)

                        # get intersecting polygons with patch and create energy mask
                        touched_gdf = label_df[label_df.geometry.intersects(patch_polygon_)]
                        if touched_gdf.shape[0] > 0:
                            combined_geom = touched_gdf.geometry.unary_union
                            minx_poly, miny_poly, maxx_poly, maxy_poly = combined_geom.envelope.bounds
                            minx_patch, miny_patch, maxx_patch, maxy_patch = patch_polygon_.bounds
                            minx, miny, maxx, maxy = min(minx_patch, minx_poly), min(miny_patch, miny_poly), \
                                max(maxx_patch, maxx_poly), max(maxy_patch, maxy_poly)
                            poly_patch_window = img.window(minx, miny, maxx, maxy)
                            temp_window_arr = img.read(window=poly_patch_window, boundless=True, fill_value=0)
                            _, row_, col_ = temp_window_arr.shape
                            patch_transform2 = rasterio.windows.transform(poly_patch_window, img_transform)
                            energy_map = interior_polygons_with_euc(touched_gdf, height=col_, width=row_,
                                                                    transform=patch_transform2,
                                                                    exterior_dist=self.exterior_dist)

                            # read energy map within patch window and write

                            with MemoryFile() as memfile:
                                with memfile.open(driver='GTiff', height=row_, width=col_, count=1,
                                                  dtype=energy_map.dtype,
                                                  crs=img.crs, transform=patch_transform2) as dataset:
                                    dataset.write(energy_map, 1)
                                    patch_window3 = dataset.window(minx_patch, miny_patch, maxx_patch, maxy_patch)
                                    patch_energy_map = dataset.read(window=patch_window3)
                        else:
                            patch_energy_map = np.expand_dims(np.zeros((self.patch_size,
                                                                        self.patch_size)).astype(np.uint8), axis=0)
                        filename_m = os.path.join(*[self.out_folder, self.int_euc_folder_name, filename_])
                        patch_transform_ = rasterio.windows.transform(patch_window2, img_transform)
                        with rasterio.open(filename_m, 'w', driver='GTiff', height=self.patch_size,
                                           width=self.patch_size, count=1, dtype=patch_energy_map.dtype, crs=img.crs,
                                           transform=patch_transform_) as dst:
                            dst.write(patch_energy_map)
                    # energy mask
                    if self.energy_mask:
                        patch_window2 = Window(col_off=col_off, row_off=row_off,
                                               width=self.patch_size,
                                               height=self.patch_size)
                        patch_polygon_ = window_to_polygon(img_transform, patch_window2)

                        # get intersecting polygons with patch and create energy mask
                        touched_gdf = label_df[label_df.geometry.intersects(patch_polygon_)]
                        if touched_gdf.shape[0] > 0:
                            combined_geom = touched_gdf.geometry.unary_union
                            minx_poly, miny_poly, maxx_poly, maxy_poly = combined_geom.envelope.bounds
                            minx_patch, miny_patch, maxx_patch, maxy_patch = patch_polygon_.bounds
                            minx, miny, maxx, maxy = min(minx_patch, minx_poly), min(miny_patch, miny_poly), \
                                max(maxx_patch, maxx_poly), max(maxy_patch, maxy_poly)
                            poly_patch_window = img.window(minx, miny, maxx, maxy)
                            temp_window_arr = img.read(window=poly_patch_window, boundless=True, fill_value=0)
                            _, row_, col_ = temp_window_arr.shape
                            patch_transform2 = rasterio.windows.transform(poly_patch_window, img_transform)
                            energy_map = energy_levels_with_distance(touched_gdf, height=col_, width=row_,
                                                                     transform=patch_transform2,
                                                                     level_dist=self.level_dist)

                            # read energy map within patch window and write
                            with MemoryFile() as memfile:
                                with memfile.open(driver='GTiff', height=row_, width=col_, count=1,
                                                  dtype=energy_map.dtype,
                                                  crs=img.crs, transform=patch_transform2) as dataset:
                                    dataset.write(energy_map, 1)
                                    patch_window3 = dataset.window(minx_patch, miny_patch, maxx_patch, maxy_patch)
                                    patch_energy_map = dataset.read(window=patch_window3)
                        else:
                            patch_energy_map = np.expand_dims(np.zeros((self.patch_size,
                                                                        self.patch_size)).astype(np.uint8), axis=0)
                        filename_m = os.path.join(*[self.out_folder, f'energy_mask{self.level_dist}', filename_])
                        patch_transform_ = rasterio.windows.transform(patch_window2, img_transform)
                        with rasterio.open(filename_m, 'w', driver='GTiff', height=self.patch_size,
                                           width=self.patch_size, count=1, dtype=patch_energy_map.dtype, crs=img.crs,
                                           transform=patch_transform_) as dst:
                            dst.write(patch_energy_map)

                else:
                    valid_patch = 3
            else:
                data_per, data_area, valid_patch, filename = 0, 0, 4, None
            filename = None
            self.patch_rows.append([self.start_index, data_per, data_mask.sum(), valid_patch, filename])
            self.patch_geo.append(patch_polygon)
            self.start_index += 1

    def run_datagen_pipeline(self):
        image_list = sorted(glob.glob(os.path.join(self.image_dir, '*.' + self.image_format)))
        label_list = sorted(glob.glob(os.path.join(self.label_dir, '*.' + self.label_format)))
        print(f'Number of images : {len(image_list)} and Number of labels : {len(label_list)}')
        assert len(image_list) == len(label_list), f'Length of images and labels files are not same'

        for image_path, label_path in tqdm(zip(image_list, label_list), total=len(image_list), desc='Processing images',
                                           unit='image'):
            if hasattr(self, 'subset_files'):
                if os.path.basename(image_path) not in self.subset_files:
                    continue
            self.patch_gen(image_path=image_path, label_path=label_path)
            print(f'Labels created for {image_path} and {label_path}')

        # create a shapefile of bbox of patches fixme try to write lists to files in loop (it reduces memory)
        df = pd.DataFrame(self.patch_rows)
        df.columns = ["patch_id", "data_percentage", "data_area", 'valid_patch', 'filename']
        geodf = gpd.GeoDataFrame(df,
                                 geometry=self.patch_geo).set_crs(
            gpd.read_file(label_path).crs)  # fixme error while adding crs but not required
        geodf.to_file(self.patch_boundaries, driver='ESRI Shapefile')

        # create final coco format json file
        if self.coco_labels or self.yolo_labels or self.yolo_binary:
            final_coco_dict = custom_coco_from_polygons(self.final_image_dicts, self.final_annotation_dicts,
                                                        dataset_name=self.coco_category_dict)
            save_coco_json(self.coco_annotations_file, final_coco_dict)

            # create yolo format labels
            if self.yolo_labels:
                convert_coco_json(json_dir=self.out_folder, use_segments=True,
                                  save_dir=self.yolo_label_folder)
                cls_names = get_class_names(self.coco_annotations_file)
                create_yml_file(data_path=os.path.abspath(self.out_folder), n_classes=len(cls_names), names=cls_names)

            if self.yolo_binary:
                convert_coco_json(json_dir=self.out_folder, use_segments=True,
                                  save_dir=self.yolo_binary_folder, binary_dataset=True)
                cls_names = ["building"]
                create_yml_file(data_path=os.path.abspath(self.out_folder), n_classes=len(cls_names), names=cls_names)
