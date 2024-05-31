"""
Code for deteciton of objects in images using object detection models.
Currently implemented for YOLOv8.
"""
import json
import os
import pandas as pd
import rasterio
import torch
from pipelines import fcn_prediction, base_class
from ultralytics import YOLO
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from utils.rasterio_utils import get_transform


class ObjectDetection(fcn_prediction.FCNPredict):
    def __init__(self, image_dir, weights_path, num_classes, patch_size, stride, out_dir,
                 pred_batch_size=1, keyword='yolo', classifier_path=None, dt_geojson=None, use_dinov2cls=False):
        super().__init__(out_dir=out_dir, keyword=keyword)
        self.image_dir = image_dir
        self.pred_batch_size = pred_batch_size
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stride = stride
        self.out_dir = out_dir
        self.base_class = base_class.BaseClass(self.image_dir)
        self.image_list = self.base_class.create_image_list()
        self.label_list = self.base_class.create_label_list()
        self.model = self.load_model()
        self.poly_id = 0
        self.keyword = keyword
        self.coco_annotation = []
        self.dataframes = []
        self.image_id = 0
        self.classifier_path = classifier_path
        self.use_dinov2cls = use_dinov2cls
        self.score_attr = 'score'
        if dt_geojson is None:
            self.dt_geojson = os.path.join(self.out_dir, f'dt_data_{self.keyword}.geojson')
        self.dt_geojson = dt_geojson

    def load_model(self):
        model = YOLO(self.weights_path)
        return model

    def get_binary_mask(self, output):
        """Merge multiple masks into one mask"""
        batch_tensor = torch.zeros((len(output), self.patch_size, self.patch_size))
        for i in range(len(output)):
            data = output[i].masks
            if data is not None:
                patch_tensor = torch.zeros((self.patch_size, self.patch_size))
                for idx in range(data.shape[0]):
                    temp = data.data[idx][None, None, :, :]
                    temp = torch.nn.functional.interpolate(temp, size=(self.patch_size, self.patch_size),
                                                           mode='nearest').squeeze().to('cpu')
                    patch_tensor = torch.max(patch_tensor, temp)
                batch_tensor[i] = patch_tensor
        return batch_tensor

    def non_maximum_suppression_iou(self, gdf, iou_threshold=0.4):
        """
        Apply Non-Maximum Suppression based on Intersection over Union (IoU), merging smaller polygons into larger ones.
        """
        # Fix invalid geometries
        invalid_geometries = gdf[~gdf.is_valid]
        if not invalid_geometries.empty:
            gdf['geometry'] = gdf.apply(
                lambda row: row['geometry'].buffer(0) if not row['geometry'].is_valid else row['geometry'], axis=1)

        # Reset index for consistent indexing
        gdf = gdf.reset_index(drop=True)

        # Track polygons to be merged
        merge_dict = {}
        for i, poly1 in enumerate(gdf.geometry):
            for j, poly2 in enumerate(gdf.geometry):
                if i >= j or i in merge_dict or j in merge_dict:
                    # Skip checking the same pair twice or with polygons already marked for merging
                    continue
                intersection = poly1.intersection(poly2).area
                iou = intersection / min(poly1.area, poly2.area)
                if iou >= iou_threshold:
                    # Merge the smaller polygon into the larger one
                    if poly1.area < poly2.area:
                        merge_dict[i] = j
                    else:
                        merge_dict[j] = i

        # Perform the merge operation
        for smaller, larger in merge_dict.items():
            gdf.at[larger, 'geometry'] = unary_union([gdf.at[larger, 'geometry'], gdf.at[smaller, 'geometry']])

        # Drop merged polygons
        gdf = gdf.drop(index=list(merge_dict.keys()))
        return gdf

    def get_polygons_from_yolo(self, output):
        """Get polygons from yolo output"""
        batch_polygons = []
        batch_conf = []
        batch_cls = []
        for i in range(len(output)):
            data = output[i].masks
            # import ipdb; ipdb.set_trace()
            patch_polys = []
            if data is not None:
                coords = data.xy
                for idx in range(len(coords)):
                    poly = coords[idx]
                    patch_polys.append(poly)
            batch_polygons.append(patch_polys)
            batch_conf.append(output[i].boxes.conf)
            batch_cls.append(output[i].boxes.cls)
        return batch_polygons, batch_conf, batch_cls

    def predict_image_using_yolo(self, image_array, batch_pos, img_height, img_width):
        """Predict one batch of patches with tensorflow, and add result to the output prediction. """
        instance_mask = torch.zeros((img_height, img_width), dtype=torch.uint8, device='cpu')
        polygons = []
        for i in tqdm(range(0, len(image_array), self.pred_batch_size)):

            X = torch.tensor(image_array[i:i + self.pred_batch_size] / 255).float().to(self.device)
            X_pos = batch_pos[i:i + self.pred_batch_size]
            outputs = self.model(X, imgsz=(640, 640), verbose=False, save_conf=True, conf=0.3)

            # add predicted polygons to list
            poly_batch, conf_batch, cls_batch = self.get_polygons_from_yolo(outputs)

            for idx, (col, row, wi, he) in enumerate(X_pos):
                for cdx, poly in enumerate(poly_batch[idx]):
                    poly[:, 0] += col
                    poly[:, 1] += row
                    if len(poly) < 3:
                        continue
                    polygons.append([self.poly_id, conf_batch[idx][cdx].item(), cls_batch[idx][cdx].item() + 1,
                                     Polygon(poly)])
                    self.poly_id += 1
            # add predicted mask to instance mask
            binary_mask = self.get_binary_mask(outputs)
            for idx, (col, row, wi, he) in enumerate(X_pos):
                instance_mask[row:row + he, col:col + wi] = torch.max(binary_mask[idx, :he, :wi],
                                                                      instance_mask[row:row + he, col:col + wi])

        return instance_mask, polygons

    def combine_splitted_with_remaining(self, gdf):
        new_rows = []
        processed_indices = set()
        for i, row_i in gdf.iterrows():
            for j, row_j in gdf.iterrows():
                if i < j and row_i.geometry.intersects(row_j.geometry):
                    processed_indices.add(i)
                    processed_indices.add(j)
                    intersection = row_i.geometry.intersection(row_j.geometry)
                    new_row_i = row_i.copy()
                    new_row_i.geometry = row_i.geometry.difference(intersection)
                    new_rows.append(new_row_i)
                    new_row_j = row_j.copy()
                    new_row_j.geometry = row_j.geometry.difference(intersection)
                    new_rows.append(new_row_j)

        # Add remaining rows that were not part of any intersection
        for i, row in gdf.iterrows():
            if i not in processed_indices and not row.geometry.is_empty:
                new_rows.append(row)

        # Create a new GeoDataFrame from the new rows
        final_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
        return final_gdf.reset_index(drop=True)

    def transform_polygons(self, gdf, raster_path):
        with rasterio.open(raster_path) as src:
            def transform_polygon(polygon):
                transformed_coords = [
                    src.xy(row, col) for col, row in polygon.exterior.coords
                ]
                return Polygon(transformed_coords)

            def transform_multipolygon(multipolygon):
                transformed_polygons = [
                    transform_polygon(polygon) for polygon in multipolygon.geoms
                ]
                return MultiPolygon(transformed_polygons)

            def transform_geometry(geometry):
                if isinstance(geometry, Polygon):
                    return transform_polygon(geometry)
                elif isinstance(geometry, MultiPolygon):
                    return transform_multipolygon(geometry)
                else:
                    return geometry  # or raise an error if unexpected geometry type

            # Apply the transformation to each geometry
            gdf['geometry'] = gdf['geometry'].apply(transform_geometry)

        return gdf

    def predict_all_images(self):
        for idx, image_path in enumerate(self.image_list):
            patch_arrays, patch_pos, image_height, image_width = self.create_patch_array(image_path)
            output, polygons = self.predict_image_using_yolo(patch_arrays, patch_pos, image_height, image_width)

            raw_poly = gpd.GeoDataFrame(polygons, columns=['id', 'score', 'label', 'geometry'])
            step1_poly = self.non_maximum_suppression_iou(raw_poly)
            # step1_poly = self.combine_splitted_with_remaining(step1_poly)

            if len(step1_poly) == 0:
                continue
            final_poly = self.transform_polygons(step1_poly, image_path)

            if self.use_dinov2cls:
                final_poly = self.add_score_class_to_polys(image_path, final_poly)
            print(f'Number of segments in the image: {len(final_poly)}')
            if self.label_list:
                label_gdf = gpd.read_file(self.label_list[idx])
                print(f'Number of ground truth polygons: {len(label_gdf)}')

            self.dataframes.append(final_poly)
            self.image_id += 1

        # save dataframes
        dt_final_gdf = gpd.GeoDataFrame(pd.concat(self.dataframes, ignore_index=True))
        dt_final_gdf.to_file(self.dt_geojson)
