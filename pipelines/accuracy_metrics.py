"""
This module contains the code for calculating accuracy metrics from
 ground truth and detection shapefiles.
"""

from utils.ap50_utils import calculate_ap50, calculate_ap5095
import geopandas as gpd
from shapely.ops import unary_union


class AccuracyMetrics:
    def __init__(self, gt_shapefile_path, det_shapefile_path, score_attr='score', gt_class_attr='material',
                 dt_class_attr='label', gt_src_crs=4326, gt_dst_crs=32737, dt_src_crs=4326,
                 dt_dst_crs=32737, noise_area=1, metric_list=None, num_classes=2):
        if metric_list is None:
            self.metric_list = ['iou', 'ap50', 'ap5095']
            if num_classes > 2:
                self.metric_list += ['miou', 'ciou', 'map50', 'map5095', 'cap50']

        else:
            self.metric_list = metric_list
        self.gt_shapefile_path = gt_shapefile_path
        self.det_shapefile_path = det_shapefile_path
        self.score_attr = score_attr
        self.gt_class_attr = gt_class_attr
        self.dt_class_attr = dt_class_attr
        self.gt_src_crs = gt_src_crs
        self.gt_dst_crs = gt_dst_crs
        self.dt_src_crs = dt_src_crs
        self.dt_dst_crs = dt_dst_crs
        self.noise_area = noise_area
        # check if gt_shapefile_path is geodataframe else read shapefile
        if isinstance(gt_shapefile_path, gpd.GeoDataFrame):
            self.gt_df = gt_shapefile_path
        else:
            self.gt_df = gpd.read_file(gt_shapefile_path)
        self.dt_df = gpd.read_file(det_shapefile_path)
        self.gt_df = self.change_crs(self.gt_df, self.gt_src_crs, self.gt_dst_crs)
        self.dt_df = self.change_crs(self.dt_df, self.dt_src_crs, self.dt_dst_crs)
        self.dt_df = self.remove_small_objects(self.dt_df, self.noise_area)

    def change_crs(self, df, src_crs=None, dst_crs=None):
        if src_crs:
            df = df.set_crs(src_crs, allow_override=True)
        if dst_crs:
            df = df.to_crs(dst_crs)
        return df

    def remove_small_objects(self, df, area_threshold):
        print(f"Number of objects before removing small objects: {len(df)}")
        df = df[df.area > area_threshold]
        print(f"Number of objects after removing small objects: {len(df)}")
        return df

    def calculate_iou(self, gt_df, dt_df):
        # Create a union of all polygons in each GeoDataFrame
        union_gt = unary_union(gt_df['geometry'])
        union_dt = unary_union(dt_df['geometry'])

        # Calculate intersection and union of the two unions
        intersection = union_gt.intersection(union_dt).area
        union = union_gt.union(union_dt).area
        if union > 0:
            iou_score = intersection / union
        else:
            iou_score = 0

        return iou_score

    def calculate_miou(self, classes=None):
        if classes is None:
            classes = set(self.gt_df[self.gt_class_attr]).intersection(set(self.dt_df[self.dt_class_attr]))
        print(f"Classes considering for mIoU/cIoU scores: {classes}")
        iou_scores = []
        ciou_dict = {}
        for c in classes:
            gt_df = self.gt_df[self.gt_df[self.gt_class_attr] == c]
            dt_df = self.dt_df[self.dt_df[self.dt_class_attr] == c]
            _iou = self.calculate_iou(gt_df, dt_df)
            iou_scores.append(_iou)
            ciou_dict[c] = _iou

        return sum(iou_scores) / len(iou_scores), ciou_dict

    # calculate all metrics
    def calculate_all_metrics(self):
        print(f"Ground truth CRS: {self.gt_df.crs}, Detection CRS: {self.dt_df.crs}")
        print(f"Estimating {self.metric_list} metrics...")
        # confidence score for binary results
        binary_attr = 'score'
        if 'score1' in self.dt_df.columns:
            self.score_attr = 'score1'

        all_metrics = {}
        for m in self.metric_list:
            if m == 'iou':
                iou = self.calculate_iou(self.gt_df, self.dt_df)
                print(f'IoU Score: {iou}')
                all_metrics['iou'] = iou

            elif m == 'miou':
                miou, _ = self.calculate_miou()
                print(f'mIOU Score: {miou}')
                all_metrics['miou'] = miou

            elif m == 'ciou':
                _, ciou = self.calculate_miou(classes=set([1, 2, 3, 4, 5]))
                print(f'cIOU Scores: {ciou}')
                all_metrics['ciou'] = list(ciou.values())

            elif m == 'ap50':
                ap50, tp, _ = calculate_ap50(self.gt_df, self.dt_df, each_class=False, score_attr=binary_attr)
                print(f'AP50 Score: {ap50}')
                print(f'True Possitives: {tp}')
                all_metrics['ap50'] = ap50
                all_metrics['tp'] = tp

            elif m == 'ap5095':
                ap5095 = calculate_ap5095(self.gt_df, self.dt_df, each_class=False, score_attr=binary_attr)
                print(f'AP5095 Score: {ap5095}')
                all_metrics['ap5095'] = ap5095

            elif m == 'map50':
                map50, tp, cap50 = calculate_ap50(self.gt_df, self.dt_df, each_class=True, score_attr=self.score_attr,
                                                  gt_attr=self.gt_class_attr, dt_attr=self.dt_class_attr)
                print(f'mAP50 Score: {map50}')
                print(f'True Possitives: {tp}')
                all_metrics['map50'] = map50
                all_metrics['tp_c'] = tp

            elif m == 'map5095':
                map5095 = calculate_ap5095(self.gt_df, self.dt_df, each_class=True, score_attr=self.score_attr,
                                           gt_attr=self.gt_class_attr, dt_attr=self.dt_class_attr)
                print(f'mAP5095 Score: {map5095}')
                all_metrics['map5095'] = map5095
            elif m == 'cap50':
                if 'map50' not in self.metric_list:
                    _, _, cap50 = calculate_ap50(self.gt_df, self.dt_df, each_class=True, score_attr=self.score_attr,
                                                 gt_attr=self.gt_class_attr, dt_attr=self.dt_class_attr)
                print(f'cAP50 Scores: {cap50}')
                all_metrics['cap50'] = list(cap50.values())
            else:
                raise ValueError(f'Invalid metric name: {m}')
        return all_metrics
