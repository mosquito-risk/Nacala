import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pycocotools.mask import decode, frPyObjects
from scipy.ndimage import distance_transform_edt as distance, binary_erosion
from joblib import Parallel, delayed
from rasterio.features import geometry_mask
import rasterio
import geopandas as gpd
import shapely
import shapely.geometry
from concurrent.futures import ProcessPoolExecutor


def generate_binary_mask_from_polygon(img_shape, polygon):
    """
    Generate a binary mask of shape (height, width) using a list of polygons.
    Each polygon is represented as a list of [x1, y1, x2, y2, ...].
    """
    # Combine all polygons into one binary mask
    rle = frPyObjects(polygon, img_shape[0], img_shape[1])
    mask = decode(rle)
    return mask.astype(np.uint8)


def compute_ann_distance(ann, shape, distance_=True):
    # Assuming ann['segmentation'] contains the RLE representation
    if isinstance(ann['segmentation'], dict):  # Checking if the segmentation is in RLE format
        single_mask = decode(ann['segmentation'])
    elif isinstance(ann['segmentation'], list):
        single_mask = generate_binary_mask_from_polygon(shape, ann['segmentation'])

    single_mask = single_mask.squeeze().astype('uint8')
    if len(single_mask.shape) == 3:
        single_mask = np.max(single_mask, axis=2)

    if distance_:
        return distance(single_mask == 0)
    else:
        return single_mask, ann['category_id']


def single_polygon_weight(anns, shape, class_weight_dir):
    single_mask, cat_id = compute_ann_distance(anns[0], shape, distance_=False)
    single_mask = single_mask * cat_id
    weights = weights_to_classes(single_mask, class_weight_dir)
    return weights.squeeze(), single_mask.squeeze()


def border_weight_map(img_shape, anns, border_threshold=8, w0=10, sigma=5, label_list_=None, weights_=None):
    # generate single label mask
    all_polys = Parallel(n_jobs=-1)(delayed(compute_ann_distance)(ann, img_shape, distance_=False) for ann in anns)
    label_mask = np.zeros(img_shape)
    for tpl in all_polys:
        label_mask = np.maximum(label_mask, tpl[0] * tpl[1])

    # calculate weights
    all_poly_dist = Parallel(n_jobs=-1)(delayed(compute_ann_distance)(ann, img_shape, distance_=True) for ann in anns)
    all_distances = np.array(all_poly_dist)
    d1 = np.min(all_distances, axis=0).squeeze()
    d2 = np.partition(all_distances, 1, axis=0)[1].squeeze()
    weights = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
    weights = np.floor(weights).astype(np.uint8)

    # create final weight mask
    border_lines = np.where(weights >= border_threshold, 1, 0)  # object separate map
    label_after_sep = np.where(border_lines == 1, 0, label_mask)  # objects after masking with borders
    # if class_weight_dir is not None:
    label_after_sep_w = weights_to_classes(label_after_sep, label_list_, weights_)
    final_weights = np.where(label_after_sep_w != 0, label_after_sep_w, weights)
    final_weights = np.where(final_weights == 0, 1, final_weights)
    return final_weights, label_after_sep


def border_weight_map_rio(gdf, image=None, w0=10, sigma=5, height=None, width=None, transform=None, return_mask=False):
    if return_mask:
        label_mask = np.zeros((width, height), dtype='uint8')
    if image is not None:
        with rasterio.open(image) as src:
            transform = src.transform
            width = src.width
            height = src.height
    else:
        width, height = width, height
        transform = transform
    all_dist = []
    # label_mask = np.zeros((width, height), dtype='uint8')
    if isinstance(gdf, gpd.GeoDataFrame):
        for i, row in gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=transform, invert=True,
                                 out_shape=(width, height))
            dist = distance(mask == 0)
            # label_mask = np.maximum(label_mask, mask * row[label_attribute])  # material_id
            all_dist.append(dist)

            if return_mask:
                label_mask = np.maximum(label_mask, mask)

    # binary mask from coco format
    if isinstance(gdf, list):
        for row in gdf:
            if isinstance(row['segmentation'], list):
                rle = frPyObjects(row['segmentation'], height, width)
            else:
                rle = row['segmentation']
            mask = decode(rle).squeeze().astype('uint8')
            if len(mask.shape) == 3:
                mask = np.max(mask, axis=2)
            dist = distance(mask == 0)
            # label_mask = np.maximum(label_mask, mask * row['category_id'])  # material_id
            all_dist.append(dist)

            if return_mask:
                label_mask = np.maximum(label_mask, mask)

    if len(all_dist) > 1:
        all_distances = np.array(all_dist)
        d1 = np.min(all_distances, axis=0).squeeze()
        d2 = np.partition(all_distances, 1, axis=0)[1].squeeze()
        weights_org = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
        weights = np.round(weights_org).astype(np.uint8)
    else:
        weights = np.ones((width, height)).astype(np.uint8)
    # import ipdb; ipdb.set_trace()
    weights = np.where(weights == 0, 1, weights)

    if return_mask:
        return weights, label_mask
    else:
        return weights


def energy_levels(gdf, height=None, width=None, transform=None, num_levels=6):
    energy_mask = np.zeros((width, height), dtype='uint8')
    if isinstance(gdf, gpd.GeoDataFrame):
        for i, row in gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=transform, invert=True,
                                 out_shape=(width, height))
            energy_mask = np.maximum(energy_mask, mask)
            iter_, sum_ = 1, np.sum(mask)
            # remove ext_pers of pixels
            while iter_ < num_levels and sum_ > 0:
                mask = binary_erosion(mask, iterations=10)
                iter_, sum_ = iter_ + 1, np.sum(mask)
                energy_mask = np.maximum(energy_mask, mask * iter_)
    return energy_mask.astype(np.uint8)


def interior_polygons(gdf, int_pers=0.7, label_attribute='material_id'):
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf = gdf.to_crs(32737)
        geo_rows, att_rows = [], []
        for i, row in gdf.iterrows():
            new_row = row.geometry
            poly_sum, poly_sum_err = new_row.area, new_row.area
            while poly_sum_err/poly_sum > int_pers:
                new_row = shapely.buffer(new_row, -0.04)
                poly_sum_err = new_row.area
            geo_rows.append(new_row)
            att_rows.append(row[label_attribute])
        df = pd.DataFrame(att_rows, columns=[label_attribute])
        return gpd.GeoDataFrame(df, geometry=geo_rows).set_crs(32737).to_crs(4326)
    else:
        return NotImplemented


def interior_polygons_with_euc(gdf, height=None, width=None, transform=None, exterior_dist=1):
    mask_euc = np.zeros((width, height), dtype='uint8')
    org_mask = np.zeros((width, height), dtype='uint8')
    if isinstance(gdf, gpd.GeoDataFrame):
        for i, row in gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=transform, invert=True,
                                 out_shape=(width, height))
            distance_mask = distance(mask == 1)
            mask_euc = np.maximum(mask_euc, distance_mask)

    if isinstance(gdf, list):
        for row in gdf:
            if isinstance(row['segmentation'], list):
                rle = frPyObjects(row['segmentation'], height, width)
            else:
                rle = row['segmentation']
            mask = decode(rle).squeeze().astype('uint8')
            if len(mask.shape) == 3:
                mask = np.max(mask, axis=2)
            distance_mask = distance(mask == 1)
            mask_euc = np.maximum(mask_euc, distance_mask)
            org_mask = np.maximum(org_mask, mask)
    # int_mask = np.where(mask_euc > exterior_dist, 1, 0).astype(np.uint8)
    return mask_euc


def energy_levels_with_distance(gdf, height=None, width=None, transform=None, num_levels=6, level_dist=4):
    energy_mask = np.zeros((width, height), dtype='uint8')
    if isinstance(gdf, gpd.GeoDataFrame):
        for i, row in gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=transform, invert=True,
                                 out_shape=(width, height))
            energy_mask = np.maximum(energy_mask, mask)
            distance_mask = distance(mask == 1)
            iter_, sum_, exterior_dist = 1, np.sum(mask), 0
            # remove ext_pers of pixels
            while iter_ < num_levels and sum_ > 0:
                iter_, exterior_dist = iter_ + 1, exterior_dist + level_dist
                mask = np.where(distance_mask > exterior_dist, 1, 0).astype(np.uint8)
                energy_mask = np.maximum(energy_mask, mask * iter_)
                sum_ = np.sum(mask)
    return energy_mask.astype(np.uint8)

