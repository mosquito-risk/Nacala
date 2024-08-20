"""
Code for testing FCN,  and FCN-DINOv2 prediction code and accuracy metrics.
e.g. python nacala_object_detection.py --server lumi --data_set valid --keyword train --batch_size 16 --mask_decision both --dt_shp_path /scratch/project_465001005/projects/nacala/mar5/dt_data_train.shp --num_classes 5 --patch_size 512 --stride_subtract 0 --use_dinov2cls False
"""

import json
import os
import sys
import glob
import time
import geopandas as gpd
import argparse
import pandas as pd

sys.path.append('../../')
from pipelines import object_detection
from pipelines import accuracy_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data directory for predictions",
                        default="./datasets/sample/test/")
    parser.add_argument("--batch_size", type=int, help="Batch size for prediction", default=16)
    parser.add_argument("--keyword", type=str, help="keyword", default="test")
    parser.add_argument("--dt_geojson", type=str, help="Detected output from images", default=None)
    parser.add_argument("--num_classes", type=int, help="keyword", default=5)
    parser.add_argument("--patch_size", type=int, help="patch_size for prediction", default=512)
    parser.add_argument("--stride_subtract", type=int, help="patch_size", default=128)
    parser.add_argument("--use_dinov2cls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--classifier_path", type=str, help="Classifier trained on DINOv2 features",
                        default="./temp/classifier/subset80p_model.pkl")
    parser.add_argument("--weights_folder", type=str, help="Model weights directory",
                        default="./temp/yolo1/")
    args = parser.parse_args()

    params = {'keyword': args.keyword}
    params['patch_size'] = args.patch_size
    params['stride'] = params['patch_size'] - args.stride_subtract
    params['patch_size'] = args.patch_size
    params['pred_batch_size'] = args.batch_size
    params['num_classes'] = args.num_classes
    params['use_dinov2cls'] = args.use_dinov2cls

    params['image_dir'] = args.data_dir
    params['classifier_path'] = args.classifier_path
    params['weights_path'] = os.path.join(args.weights_folder, 'weights/iou_best.pt')
    params['out_dir'] = './temp'

    if args.dt_geojson is None:
        params['dt_geojson'] = os.path.join(params['out_dir'], f'dt_data_{params["keyword"]}.geojson')

    # print parameters using json
    print(json.dumps(params, indent=4))

    # create single dataframe from all ground truth geojson files
    files = glob.glob(f'{params["image_dir"]}/*.geojson')
    dataframesList = []
    for file in files:
        _df = gpd.read_file(file)
        dataframesList.append(_df)
    df = gpd.GeoDataFrame(pd.concat(dataframesList, ignore_index=True))
    print(f"Number of objects in the ground truth: {len(df)}")

    # Create prediction object and get shapefile of predictions
    start = time.time()

    predict_obj = object_detection.ObjectDetection(**params)
    predict_obj.predict_all_images()
    end = time.time()
    pred_time = round((end - start) / 60, 2)
    print(f"Time taken for prediction: {pred_time}")

    # get accuracy metrics
    dt_shp = gpd.read_file(params['dt_geojson'])
    noise_area = 1
    acc_classes = 5
    acc_object = accuracy_metrics.AccuracyMetrics(gt_shapefile_path=df,
                                                  det_shapefile_path=dt_shp,
                                                  noise_area=noise_area,
                                                  gt_class_attr="mater_id",
                                                  num_classes=acc_classes)
    acc_metrics = acc_object.calculate_all_metrics()
    print(f"Accuracy metrics of {args.weights_folder}: \n")
    print(json.dumps(acc_metrics, indent=4))

    # save params as json
    end = time.time()
    total_time = round((end - start) / 60, 2)
    params['pred_time'] = pred_time
    params['total_time'] = total_time
    with open(f'./temp/params_predict_{params["keyword"]}.json', 'w') as file:
        json.dump(params, file, indent=4)
