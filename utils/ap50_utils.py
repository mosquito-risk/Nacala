"""
Utility functions for AP50 and AP50-95 metrics using shapefiles

Example usage:
# Load the shapefiles
gt_shapefile_path = '../data/temp2.shp'
det_shapefile_path = '../data/gt_data.shp'

#1. caclculate ap50
ap50_score = calculate_ap50(gt_shapefile_path, det_shapefile_path, each_class=False, score_attr='score')
print(ap50_score)

#2. calculate mAP50
ap50_score = calculate_ap50(gt_shapefile_path, det_shapefile_path, each_class=True, gt_attr='mater_id',
                            dt_attr='label', score_attr='score')
print(ap50_score)

#3. calculate ap5095
print(calculate_ap5095(gt_shapefile_path, det_shapefile_path, score_attr='score'))

#4. calculate mAP5095
print(calculate_ap5095(gt_shapefile_path, det_shapefile_path, score_attr='score', each_class=True, gt_attr='mater_id',
                       dt_attr='label'))
"""

import geopandas as gpd
import numpy as np
import pandas as pd


def calculate_iou(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    iou = intersection / union if union != 0 else 0
    return iou


def average_precision(gt_dataframe, dt_dataframe, score_atrr='score', iou_thresh=0.50, auc_method='101point',
                      pr_list=False):
    # reset index for safe indices
    gt_dataframe, dt_dataframe = gt_dataframe.reset_index(drop=True), dt_dataframe.reset_index(drop=True)

    # Step-1: Sort the detections based on the score in descending order
    assert score_atrr is not None, f"Score attribute cannot be {score_atrr}"
    assert score_atrr in dt_dataframe.columns, (f"Score attribute-{score_atrr} not in list of"
                                                f" columns {dt_dataframe.columns}")
    dt_dataframe = dt_dataframe.sort_values(by=score_atrr, ascending=False).reset_index(drop=True)

    # Step-2 Calculate maximum IoU between ground truth polygon and all intersected predicted polygons
    ious = []
    used_gt_ids = set()
    for det_idx, det_poly in dt_dataframe.iterrows():
        best_iou = 0
        best_gt_id, best_det_id = None, None
        # find all intersecting polygons
        intersecting_polygons = gt_dataframe[gt_dataframe.intersects(det_poly.geometry)]
        if intersecting_polygons.shape[0] > 0:
            for gt_idx, gt_poly in intersecting_polygons.iterrows():
                # Skip if this ground truth is already used (
                if gt_idx in used_gt_ids:
                    continue
                iou = calculate_iou(det_poly.geometry, gt_poly.geometry)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_id, best_det_id = gt_idx, det_idx

            if best_iou >= iou_thresh:
                ious.append({'det_idx': best_det_id, 'gt_idx': best_gt_id, 'iou': best_iou})
                used_gt_ids.add(best_gt_id)  # Mark this ground truth as used
        else:
            pass
            # if the detected polygon is not having intersecting ground truth or IoU between less than that,
            # we are not storing that information. Added as False Possitives (check line 62)

    # Step-3: Prepare True Possitives and False Possitives
    tp = np.zeros(len(dt_dataframe))
    fp = np.zeros(len(dt_dataframe))
    scores = np.zeros(len(dt_dataframe))
    gt_matched = set()

    # Assign TP or FP to detections
    # Sort the IoUs to match the highest ones first
    ious.sort(key=lambda x: x['iou'], reverse=True)
    for match in ious:
        assert match['gt_idx'] not in gt_matched
        tp[match['det_idx']] = 1
        gt_matched.add(match['gt_idx'])
        scores[match['det_idx']] = dt_dataframe.loc[match['det_idx'], score_atrr]

    # False positives for unmatched detections
    for det_idx in range(len(dt_dataframe)):
        if det_idx not in [m['det_idx'] for m in ious]:
            fp[det_idx] = 1
            scores[det_idx] = dt_dataframe.loc[det_idx, score_atrr]

    # Step-4: Calculate precision and recall
    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recall = tp_cum / len(gt_dataframe)
    precision = tp_cum / (tp_cum + fp_cum)

    # Rule-1: Avoid division by zero
    precision = np.where(np.isnan(precision), 0, precision)
    # pr_curve(precision, recall)

    # Step-5: Sort by scores in descending order to calculate precision and recall correctly
    sorted_indices = np.argsort(-scores)
    precision = precision[sorted_indices]
    recall = recall[sorted_indices]

    # Rule-2: The precision for recall r is the maximum precision obtained for any recall r' >= r
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # pr_curve(precision, recall)
    # Step-7: Calculate AP using the trapezoidal rule correctly methods in ['trapz', '11point', '101point']
    if auc_method == 'trapz':
        ap50 = np.trapz(precision, recall)
    elif auc_method == '11point':
        # 11-point interpolation for average precision calculation
        ap50 = 0.0
        for t in np.linspace(0, 1, 11):  # 11 equally spaced points
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap50 += p / 11
    elif auc_method == '101point':
        # 11-point interpolation for average precision calculation
        ap50 = 0.0
        for t in np.linspace(0, 1, 101):  # 101 equally spaced points
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap50 += p / 101
    else:
        raise NotImplementedError

    if pr_list:
        return [precision, recall]
    else:
        return ap50, np.sum(tp)


def calculate_ap50(gt_shapefile, det_shapefile, iou_threshold=0.50, each_class=False, gt_attr=None,
                   dt_attr=None, score_attr=None, pr_list=False):
    # Load shapefiles and assign coordinate system (shapefiles are not having coordinate system)
    if not isinstance(gt_shapefile, gpd.GeoDataFrame):
        gt_gdf = gpd.read_file(gt_shapefile).to_crs(4326)
    else:
        gt_gdf = gt_shapefile.copy()
    if not isinstance(det_shapefile, gpd.GeoDataFrame):
        det_gdf = gpd.read_file(det_shapefile).set_crs(4326)
    else:
        det_gdf = det_shapefile.copy()

    # calculate ap50 score for each class
    ap50_list = []
    ap50_dict = {}
    true_possitives = 0
    if each_class:
        assert gt_attr is not None and dt_attr is not None
        # Get unique classes from ground truth and detection shapefiles
        classes = sorted(set(gt_gdf[gt_attr]).union(set(det_gdf[dt_attr])))
        # print("Classes in the data: ", classes)

        # loop through each class for calculating ap50 score
        for cls in classes:
            # Filter ground truth and detections for the current class
            gt_gdf_cls = gt_gdf[gt_gdf[gt_attr] == cls].reset_index()
            det_gdf_cls = det_gdf[det_gdf[dt_attr] == cls].reset_index()
            ap50, tp = average_precision(gt_gdf_cls, det_gdf_cls, score_atrr=score_attr, iou_thresh=iou_threshold)
            ap50_list.append(ap50)
            true_possitives += tp
            ap50_dict[cls] = ap50
        # print(f"mAP50 score: {np.mean(ap50_list)}")
        # print(f"Total True Posstives: {true_possitives}")
    else:
        if pr_list:
            pr_data = average_precision(gt_gdf, det_gdf, score_atrr=score_attr, iou_thresh=iou_threshold,
                                        pr_list=pr_list)
        else:
            ap50, tp = average_precision(gt_gdf, det_gdf, score_atrr=score_attr, iou_thresh=iou_threshold)
            ap50_list.append(ap50)
            true_possitives += tp
            # print(f"AP{str(iou_threshold)[2:]} score: {ap50}")
            # print(f"Total True Posstives: {true_possitives}")
    if pr_list:
        return pr_data
    else:
        return np.mean(ap50_list), true_possitives, ap50_dict


def calculate_ap5095(gt_shapefile, det_shapefile, each_class=False, gt_attr=None, dt_attr=None, score_attr='score'):
    iou_list = np.arange(0.5, 1, 0.05)
    aps = []
    for iou in iou_list:
        scores, _, _ = calculate_ap50(gt_shapefile, det_shapefile, iou_threshold=iou, each_class=each_class,
                                score_attr=score_attr, gt_attr=gt_attr, dt_attr=dt_attr)
        aps.append(scores)
    return np.mean(aps)
