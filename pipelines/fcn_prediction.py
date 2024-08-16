import os
import pickle

import torch
import json
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from itertools import product
from rasterio.features import shapes, geometry_mask
from rasterio.windows import Window
from pipelines import base_class, models_class
from shapely.ops import unary_union
from shapely.geometry import mapping
from rasterstats import zonal_stats

from pipelines.models.load_models import load_model
from pipelines import dinv2_featuregen, cnn_datagen
from utils.different_labels import watershed_image, single_output_from_multihead
import networkx as nx
import scipy.stats as stats


class FCNPredict(models_class.ModelsClass):
    def __init__(self, weights_path=None, model_name=None, image_dir=None, patch_size=1792, stride=1664,
                 pred_batch_size=1, num_classes=2, channels=3, out_dir=None, save_raster=False,
                 save_shapefile=True, keyword='test', classifier_path=None, dt_coco_path=None,
                 dt_geojson=None, energy_levels=False, mask_decision="both", head_size='n', loss_type=None,
                 label_dir=None, use_dinov2cls=False, label_from=1):
        super().__init__(model_name=model_name)
        self.model = None
        self.image_list = None
        self.label_list = None
        self.save_raster = save_raster
        self.save_shapefile = save_shapefile
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.weights_path = weights_path
        self.patch_size = patch_size
        self.stride = stride
        self.model_name = model_name
        self.channels = channels
        self.num_classes = num_classes
        self.pred_batch_size = pred_batch_size
        self.prediction_operator = "MAX"  # keep MAX because not trying to have overlapping predictions
        self.band_list = (1, 2, 3)
        self.out_dir = out_dir
        self.base_class = base_class.BaseClass(self.image_dir, self.label_dir)
        self.device = self.base_class.get_device()
        self.cnn_datagen = cnn_datagen.DataProcessingPipeline()
        self.keyword = keyword
        self.coco_annotation = []
        self.dataframes = []
        self.image_id = 0
        self.energy_levels = energy_levels
        self.classifier_path = classifier_path
        self.binary_label = False
        self.mask_decision = mask_decision
        self.head_size = head_size
        self.loss_type = loss_type
        self.label_from = label_from
        self.use_dinov2cls = use_dinov2cls
        if dt_coco_path is None:
            self.dt_coco_path = os.path.join(self.out_dir, f'dt_coco_{self.keyword}.json')
        else:
            self.dt_coco_path = dt_coco_path
        if dt_geojson is None:
            self.dt_geojson = os.path.join(self.out_dir, f'dt_data_{self.keyword}.geojson')
        else:
            self.dt_geojson = dt_geojson
        if self.weights_path:
            if self.loss_type == 'cross_entropy_cls':
                classes2 = 6
            else:
                classes2 = None
            self.model = load_model(self.model_name, self.device, channels=3, num_classes=self.num_classes,
                                    pre_trained_weight=self.weights_path, patch_size=self.patch_size,
                                    head_size=self.head_size, num_classes2=classes2)

    def load_classifier(self, classifier_path):
        with open(classifier_path, 'rb') as file:
            return pickle.load(file)

    def add_classifier_score_to_polys(self, polys, score_array, label_array):
        # create temp image for using rasterio window operations
        polys['score1'] = score_array
        polys['label'] = label_array
        return polys

    def add_score_class_to_polys(self, image_path, polys):
        # create batch for DINOv2 model input
        arg_list = self.cnn_datagen.create_window_per_polygon(image_path, polys)
        image_array, mask_array, _ = self.cnn_datagen.create_batch(arg_list, image_path)
        dinov2_features = dinv2_featuregen.DINOv2FeatureGen(image_array, mask_array).get_features()
        classifier = self.load_classifier(self.classifier_path)
        proba = classifier.predict_proba(dinov2_features)
        class_array = np.argmax(proba, axis=1) + 1  # class always start from 1
        score_array = np.amax(proba, axis=1)
        # add classifier's score and class to polygons
        polys = self.add_classifier_score_to_polys(polys, score_array, class_array)
        return polys

    def get_patch_offsets(self, rasterio_image, stride):
        """
        Get a list of patch offsets based on image size, patch size and stride.
        """
        # Create iterator of all patch offsets, as tuples (x_off, y_off)
        patch_offsets = list(product(range(0, rasterio_image.width, stride), range(0, rasterio_image.height, stride)))
        return patch_offsets

    def create_patch_array(self, image_path, image_bounds=False):
        """
        Create a numpy array of patches from a rasterio image and a list of patch offsets.
        """
        # Get list of patch offsets to predict for this image (use bounds argument for subset of image)
        img = rasterio.open(image_path, tiled=True, blockxsize=256, blockysize=256)
        img_height, img_width = img.height, img.width
        if image_bounds:
            img = rasterio.open(image_path, tiled=True, blockxsize=256, blockysize=256, bounds=image_bounds)
        offsets = self.get_patch_offsets(img, self.stride)

        # Create numpy array of patches
        batch_pos = []
        big_window = Window(0, 0, img.width, img.height)
        patch_array = np.zeros((len(offsets), self.channels, self.patch_size, self.patch_size), dtype=np.float32)
        for i, (col_off, row_off) in enumerate(offsets):
            patch_window = Window(col_off=col_off, row_off=row_off, width=self.patch_size,
                                  height=self.patch_size).intersection(big_window)
            temp_im = img.read(self.band_list, window=patch_window)
            patch_array[i, :, :patch_window.height, :patch_window.width] = temp_im
            batch_pos.append((patch_window.col_off, patch_window.row_off, patch_window.width, patch_window.height))
        return patch_array, batch_pos, img_height, img_width

    def predict_image(self, image_fp, loss_type=None):
        # predict image and add probabilities to mask
        image_array, batch_pos, img_height, img_width = self.create_patch_array(image_fp)
        mask = torch.zeros((self.num_classes, img_height, img_width), dtype=torch.float32, device='cpu')

        for i in tqdm(range(0, len(image_array), self.pred_batch_size)):
            X = torch.tensor(image_array[i:i + self.pred_batch_size]).float().to(self.device)
            X_pos = batch_pos[i:i + self.pred_batch_size]
            mask = self.predict_and_replace(X, X_pos, mask, self.prediction_operator, self.num_classes,
                                            loss_type=loss_type)

        return mask.detach().cpu().numpy()

    def add_fcn_score_to_polys(self, dataframe, score_array, meta):
        # Convert the scores array to a Rasterio dataset
        scores_dataset = rasterio.io.MemoryFile().open(
            driver='GTiff',
            height=score_array.shape[0],
            width=score_array.shape[1],
            count=1,
            dtype=score_array.dtype,
            crs=meta['crs'],
            transform=meta['transform']
        )
        scores_dataset.write(score_array, 1)

        # Calculate mean probabilities for each polygon
        results = []
        for _, row in tqdm(dataframe.iterrows()):
            # Rasterize the polygon
            geom = [row['geometry']]
            mask = geometry_mask(geom, transform=scores_dataset.transform, invert=True,
                                 out_shape=scores_dataset.shape)
            masked_data = score_array[mask]
            mean_prob = np.mean(masked_data)
            results.append(mean_prob)
        scores_dataset.close()
        return results

    def add_fcn_score_to_polys_v2(self, dataframe, score_array, meta):
        """
        New version to speed up the process
        """
        # Convert dataframe geometries to GeoJSON-like format
        polygons = [mapping(geom) for geom in dataframe['geometry']]

        # Perform zonal statistics to calculate mean within each polygon
        stats = zonal_stats(polygons, score_array, affine=meta['transform'], stats='mean')

        # Extract mean values from zonal statistics results
        results = [stat['mean'] for stat in stats]

        return results

    def vectorise_image(self, binary_array, metadata, binary_label=False):
        results = (
            {'properties': {'label': v}, 'geometry': s}
            for i, (s, v) in enumerate(
            shapes(binary_array, mask=None, transform=metadata['transform']))
        )
        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms)
        gdf = gdf[gdf['label'] != 0].reset_index(drop=True)
        if binary_label:
            gdf['label'] = 1
        gdf.crs = metadata['crs']
        return gdf

    def output_metadata(self, image_path):
        meta = rasterio.open(image_path).meta
        meta.update(dtype='uint8')
        meta.update(count=1)
        return meta

    def save_output(self, output, image_path):
        """Save output as raster and shapefile"""
        raster_path = os.path.join(self.out_dir, os.path.basename(image_path).replace('.tif', '_mask.tif'))
        meta = self.output_metadata(image_path)
        with rasterio.open(raster_path, 'w', **meta) as dst:
            dst.write(output.astype(np.uint8), 1)

    def merge_small_polygons_old(self, dataframe):
        sindex = dataframe.sindex
        new_polygons = []
        visited_indices = set()

        for index, polygon in dataframe.iterrows():
            if index in visited_indices:
                continue
            # Find neighbors
            possible_neighbors = list(sindex.intersection(polygon.geometry.bounds))
            neighbors = [idx for idx in possible_neighbors if
                         dataframe.geometry[idx].intersects(polygon.geometry) and idx != index]

            # Merge with neighbors if any
            merged = polygon.geometry
            for neighbor_index in neighbors:
                neighbor_polygon = dataframe.geometry[neighbor_index]
                if neighbor_polygon.area > merged.area:
                    # Take attributes from the larger polygon
                    polygon = dataframe.iloc[neighbor_index]
                merged = unary_union([merged, neighbor_polygon])
                visited_indices.add(neighbor_index)

            new_polygon = polygon.copy()
            new_polygon.geometry = merged
            new_polygons.append(new_polygon)
            visited_indices.add(index)
        return gpd.GeoDataFrame(new_polygons)

    def merge_small_polygons(self, dataframe):
        sindex = dataframe.sindex
        G = nx.Graph()

        for index, polygon in dataframe.iterrows():
            # Add nodes
            G.add_node(index, polygon=polygon)

            # Find and add edges to neighbors
            possible_neighbors = list(sindex.intersection(polygon.geometry.bounds))
            neighbors = [idx for idx in possible_neighbors if
                         dataframe.geometry[idx].intersects(polygon.geometry) and idx != index]

            for neighbor_index in neighbors:
                G.add_edge(index, neighbor_index)

        new_polygons = []

        # Find connected components and merge them
        for component in nx.connected_components(G):
            merged_polygon = None
            largest_area = 0
            largest_polygon_data = None

            for idx in component:
                polygon = G.nodes[idx]['polygon']
                if polygon.geometry.area > largest_area:
                    largest_area = polygon.geometry.area
                    largest_polygon_data = polygon

                if merged_polygon is None:
                    merged_polygon = polygon.geometry
                else:
                    merged_polygon = unary_union([merged_polygon, polygon.geometry])

            new_polygon = largest_polygon_data.copy()
            new_polygon.geometry = merged_polygon
            new_polygons.append(new_polygon)

        return gpd.GeoDataFrame(new_polygons)

    def predict_multi_output(self, image_fp, num_outputs=2, mask_decision="both"):
        """
        Predict image and return class and score arrays
        :param image_fp: file path of the image
        :param num_outputs: number of outputs from model
        :param mask_decision: output decision "both" or "mask1" or "mask2"
        "both" returs the final mask by taking mask as input of individual mask
        "mask1" returns only mask1 and "mask2" returns only mask2
        :return: class_array, score_array
        """
        image_array, batch_pos, img_height, img_width = self.create_patch_array(image_fp)
        masks = [torch.zeros((self.num_classes, img_height, img_width), dtype=torch.float32, device='cpu')
                 for _ in range(num_outputs)]

        for i in tqdm(range(0, len(image_array), self.pred_batch_size)):
            X = torch.tensor(image_array[i:i + self.pred_batch_size]).float().to(self.device)
            X_pos = batch_pos[i:i + self.pred_batch_size]
            masks = self.predict_and_replace_multi(X, X_pos, masks, self.prediction_operator)

        if mask_decision == "both":
            # write function for combining masks
            mask1 = masks[0].detach().cpu().numpy()
            mask2 = masks[1].detach().cpu().numpy()
            if mask1.shape[0] == mask2.shape[0] == self.num_classes == 1:
                class_array1 = np.where(mask1 >= 0.5, 1, 0).squeeze()
                class_array2 = np.where(mask2 >= 0.5, 1, 0).squeeze()
                score_array = mask1.squeeze()  # only score array from mask1 considered

            elif mask1.shape[0] == mask2.shape[0] == self.num_classes == 2:
                class_array1 = np.argmax(mask1, axis=0).astype(np.uint8)
                class_array2 = np.argmax(mask2, axis=0).astype(np.uint8)
                score_array = np.amax(mask1, axis=0)
            else:
                raise ValueError(f"Invalid mask shapes: {mask1.shape}, {mask2.shape}, {self.num_classes}")
            class_array = single_output_from_multihead(class_array1, class_array2)


        elif mask_decision == "mask1" or mask_decision == "mask2":
            # fixme: not implemented for single logit
            mask = masks[int(mask_decision[-1]) - 1].detach().cpu().numpy()
            class_array = np.argmax(mask, axis=0).astype(np.uint8)
            score_array = np.amax(mask, axis=0)
        else:
            raise ValueError(f"Invalid mask_decision: {mask_decision}")

        return class_array, score_array

    def predict_multi_output_cls(self, image_fp, mask_decision="both"):
        """
        Predict image and return class and score arrays
        :param image_fp: file path of the image
        :param num_outputs: number of outputs from model
        :param mask_decision: output decision "both" or "mask1" or "mask2"
        "both" returs the final mask by taking mask as input of individual mask
        "mask1" returns only mask1 and "mask2" returns only mask2
        :return: class_array, score_array
        """
        class_list = [1, 6]
        image_array, batch_pos, img_height, img_width = self.create_patch_array(image_fp)
        masks = [torch.zeros((_, img_height, img_width), dtype=torch.float32, device='cpu')
                 for _ in class_list]

        for i in tqdm(range(0, len(image_array), self.pred_batch_size)):
            X = torch.tensor(image_array[i:i + self.pred_batch_size]).float().to(self.device)
            X_pos = batch_pos[i:i + self.pred_batch_size]
            masks = self.predict_and_replace_multi_cls(X, X_pos, masks, self.prediction_operator)

        class_array1 = masks[0].numpy().squeeze()
        mask2 = masks[1].numpy()
        class_array2 = np.argmax(mask2, axis=0).astype(np.uint8)
        score_array = np.amax(mask2, axis=0)

        if mask_decision == "both":
            class_array2_binary = (class_array2 > 0).astype(np.uint8)
            class_array = single_output_from_multihead(class_array1, class_array2_binary)

        elif mask_decision == "both2":
            print("Using both2")
            class_array2_binary = (class_array2 > 0).astype(np.uint8)
            class_array = single_output_from_multihead(class_array2_binary, class_array1)
            final_label = np.zeros_like(class_array)
            for i in range(1, np.unique(class_array).shape[0]):
                segment_label = stats.mode(class_array2[(class_array == i) & (class_array2 != 0)]).mode
                final_label = np.where(class_array == i, segment_label, final_label)

        else:
            raise ValueError(f"Invalid mask_decision: {mask_decision}")

        return class_array, score_array, class_array2

    def predict_multi_output_cls3(self, image_fp, mask_decision="both"):
        """
        Predict image and return class and score arrays
        :param image_fp: file path of the image
        :param num_outputs: number of outputs from model
        :param mask_decision: output decision "both" or "mask1" or "mask2"
        "both" returs the final mask by taking mask as input of individual mask
        "mask1" returns only mask1 and "mask2" returns only mask2
        :return: class_array, score_array
        """
        class_list = [6, 6]
        image_array, batch_pos, img_height, img_width = self.create_patch_array(image_fp)
        masks = [torch.zeros((_, img_height, img_width), dtype=torch.float32, device='cpu')
                 for _ in class_list]

        for i in tqdm(range(0, len(image_array), self.pred_batch_size)):
            X = torch.tensor(image_array[i:i + self.pred_batch_size]).float().to(self.device)
            X_pos = batch_pos[i:i + self.pred_batch_size]
            masks = self.predict_and_replace_multi_cls3(X, X_pos, masks, self.prediction_operator)

        # class_array1 = masks[0].numpy().squeeze()
        mask1 = masks[0].numpy()
        class_array0 = np.argmax(mask1, axis=0).astype(np.uint8)
        class_array1 = (class_array0 > 0).astype(np.uint8)
        mask2 = masks[1].numpy()
        class_array2 = np.argmax(mask2, axis=0).astype(np.uint8)
        score_array = np.amax(mask1, axis=0)
        if self.label_from == 1:
            final_class_array = class_array0
        elif self.label_from == 2:
            final_class_array = class_array2
        elif self.label_from == 3:
            final_class_array = np.where(class_array2 == 0, class_array0, class_array2)
        else:
            raise ValueError(f"Invalid label_from: {self.label_from}")

        if mask_decision == "both":
            class_array2_binary = (class_array2 > 0).astype(np.uint8)
            class_array = single_output_from_multihead(class_array1, class_array2_binary)

        elif mask_decision == "both2":
            print("Using both2")
            class_array2_binary = (class_array2 > 0).astype(np.uint8)
            class_array = single_output_from_multihead(class_array2_binary, class_array1)
            final_label = np.zeros_like(class_array)
            for i in range(1, np.unique(class_array).shape[0]):
                segment_label = stats.mode(class_array2[(class_array == i) & (class_array2 != 0)]).mode
                final_label = np.where(class_array == i, segment_label, final_label)

        else:
            raise ValueError(f"Invalid mask_decision: {mask_decision}")

        return class_array, score_array, final_class_array

    def predict_all_images(self):
        # loop through all images
        self.image_list = self.base_class.create_image_list()
        self.label_list = self.base_class.create_label_list()
        for idx, image in enumerate(self.image_list):
            meta = self.output_metadata(image)
            # predict image (if image is big then it required to implement with image_bounds)
            if self.model_name == 'unet_2heads' or self.model_name == 'dinov2_2heads':
                if self.loss_type == 'cross_entropy_cls':
                    class_array, score_array, label_array = self.predict_multi_output_cls(image,
                                                                                          mask_decision=self.mask_decision)
                if self.loss_type == 'cross_entropy_cls3':
                    class_array, score_array, label_array = self.predict_multi_output_cls3(image,
                                                                                           mask_decision=self.mask_decision)
                else:
                    class_array, score_array = self.predict_multi_output(image, mask_decision="both")
                    self.binary_label = True
            elif self.energy_levels:
                output = self.predict_image(image, loss_type=self.loss_type)
                score_array = np.amax(output, axis=0)
                energy_map = (output >= 0.5).cumprod(0).sum(0)
                class_array = watershed_image(energy_map)
                self.binary_label = True
            else:
                output = self.predict_image(image)
                score_array = np.amax(output, axis=0)
                if self.num_classes == 1:
                    class_array = (output >= 0.5).astype(np.uint8)
                else:
                    class_array = np.argmax(output, axis=0).astype(np.uint8)

            # save class array as rasterio image
            filename = os.path.join(self.out_dir, os.path.basename(image))
            if self.save_raster:
                with rasterio.open(filename, 'w', **meta) as dst:
                    dst.write(class_array, 1)

            gdf = self.vectorise_image(class_array, meta, binary_label=self.binary_label)
            print(f'Number of segments in the image: {len(gdf)}')
            if self.loss_type == 'cross_entropy_cls' or self.loss_type == 'cross_entropy_cls3':
                polygons = [mapping(geom) for geom in gdf['geometry']]
                stats = zonal_stats(polygons, label_array, affine=meta['transform'], stats='majority', nodata=0)
                gdf['label'] = [stat['majority'] for stat in stats]
                print(f'Added Label to polygons')
            gdf['score'] = self.add_fcn_score_to_polys_v2(gdf, score_array, meta)
            print(f'Added FCN Score to polygons')
            gdf['score'] = self.add_fcn_score_to_polys_v2(gdf, score_array, meta)
            print(f'Added FCN Score to polygons')

            # print ground truth building polygons
            if self.label_list:
                label_gdf = gpd.read_file(self.label_list[idx])
                print(f'Number of ground truth polygons: {len(label_gdf)}')

            self.dataframes.append(gdf)
            self.image_id += 1

        # save coco annotation
        with open(self.dt_coco_path, 'w') as file:
            json.dump(self.coco_annotation, file, indent=4)

        # save dataframes
        dt_final_gdf = gpd.GeoDataFrame(pd.concat(self.dataframes, ignore_index=True))
        dt_final_gdf.to_file(self.dt_geojson)
