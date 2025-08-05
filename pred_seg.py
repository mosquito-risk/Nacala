"""
Code for testing FCN,  and FCN-DINOv2 prediction code and accuracy metrics.
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
from pipelines import fcn_cnn_prediction, fcn_prediction
from pipelines import accuracy_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name", default="unet_2heads")
    parser.add_argument("--keyword", type=str, help="Keyword for output filenames", default="test")
    parser.add_argument("--data_dir", type=str, help="Data directory for predictions",
                        default="/scratch/project_465002161/projects/Nacala/datasets/raw_data/test2")
    parser.add_argument("--weights_folder", type=str, help="Model weights directory",
                        default="/scratch/project_465002161/projects/Nacala/outputs")
    parser.add_argument("--classifier_path", type=str, help="Classifier trained on DINOv2 features",
                        default="./temp/classifier/logistic_model.pkl")
    parser.add_argument("--mask_decision", type=str, help="mask decision for multi head unet",
                        default="both")
    parser.add_argument("--dt_geojson", type=str, help="Detected output from images", default=None)
    parser.add_argument("--num_classes", type=int, help="Number of classes in seg model", default=1)
    parser.add_argument("--patch_size", type=int, help="Patch size for prediction", default=4096)
    parser.add_argument("--label_from", type=int, help="Leabel from is for multihead models 1) is for head1,"
                                                       "2) is for head2 3) is for both heads", default=1)
    parser.add_argument("--stride_subtract", type=int, help="Overlap between patches in prediction",
                        default=128)
    parser.add_argument("--use_dinov2cls", action=argparse.BooleanOptionalAction,
                        help="Use dinov2 classifier on top of binary segmentation model", default=False)
    parser.add_argument("--energy_levels", action=argparse.BooleanOptionalAction,
                        help="Using UNet model that trained for energy levels", default=False)
    parser.add_argument("--head_size", type=str, help="Size of head in 2 head model", default="n")
    parser.add_argument("--loss_type", type=str,
                        help="Special predictions for UNet 2heads with multi class (cross_entropy_cls)", default=None)
    args = parser.parse_args()

    params = {'keyword': args.keyword}
    params['patch_size'] = args.patch_size
    params['stride'] = params['patch_size'] - args.stride_subtract
    params['patch_size'] = args.patch_size
    params['model_name'] = args.model_name
    params['num_classes'] = args.num_classes
    params['energy_levels'] = args.energy_levels
    params['mask_decision'] = args.mask_decision
    params['head_size'] = args.head_size
    params['loss_type'] = args.loss_type
    params['use_dinov2cls'] = args.use_dinov2cls


    params['image_dir'] = args.data_dir
    params['classifier_path'] = args.classifier_path
    params['weights_path'] = os.path.join(args.weights_folder, args.keyword, 'best_model.pt')
    params['out_dir'] = './temp'

    if args.dt_geojson is None:
        params['dt_geojson'] = os.path.join(params['out_dir'], f'dt_{params["keyword"]}.geojson')

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
    if args.use_dinov2cls:
        predict_obj = fcn_cnn_prediction.FCNCNNPredict(**params)
    else:
        params['label_from'] = args.label_from
        predict_obj = fcn_prediction.FCNPredict(**params)
    predict_obj.predict_all_images()
    end = time.time()
    pred_time = round((end - start) / 60, 2)
    print(f"Time taken for prediction: {pred_time}")

    # get accuracy metrics
    params['dinov2_classes'] = 5
    dt_shp = gpd.read_file(params['dt_geojson'])
    noise_area = 1
    acc_object = accuracy_metrics.AccuracyMetrics(gt_shapefile_path=df,
                                                  det_shapefile_path=params['dt_geojson'],
                                                  noise_area=noise_area,
                                                  gt_class_attr="mater_id",
                                                  num_classes=params['dinov2_classes'])
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
