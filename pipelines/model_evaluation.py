"""
Evaluate model performance on test data:
currently, it takes the input of ground truth shapefiles/geojson files, model, and test images
and returns the IoU, AP50, AP50-95, and True Positives
Also, labels in coco format and predictions in coco and shapefile formats
"""
import sys
import glob
import json
import torch
import pickle
import os.path
from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage as ndi

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.coco_format_utils import coco_annotation_dict, custom_coco_from_polygons, save_coco_json
from utils.gdal_raster_utils import get_gdal_transform
from utils.rasterio_utils import bbox_to_window
from utils.prediction_utils import predict_image
from unet.models import dinov2_cls
from segmentation_models_pytorch import Unet
from pipelines.models.load_models import load_model


class evaluate:
    def __init__(self, test_image_dir, test_label_dir, num_classes=5, classifier_path=None, seg_model_name=None,
                 seg_model_weights=None):
        self.test_image_dir = test_image_dir
        self.test_label_dir = test_label_dir
        self.num_classes = num_classes
        self.classifier_path = classifier_path
        self.seg_model_name = seg_model_name
        self.seg_model_weights = seg_model_weights
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

    # Use DINOv2 model for segment classification
    def load_dinov2_model(self):
        id2label = list(range(self.num_classes))
        return dinov2_cls.Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",
                                                                        id2label=id2label,
                                                                        num_labels=self.num_classes,
                                                                        num_classes=self.num_classes,
                                                                        avg_pool=True)

    # Load classifier to run on top of dinov2 model
    def load_classifier(self):
        with open(self.classifier_path, 'rb') as file:
            return pickle.load(file)

    # load segmentation model
    def load_seg_model(self):
        model = load_model(self.seg_model_name, self.device, channels=3, num_classes=self.num_classes,
                           pre_trained_weight=self.model_weights, patch_size=self.patch_size)
    if self.model_name == 'unet':
        unet = Unet(in_channels=3, classes=2)
        if pre_trained_weight is not None:
            unet.load_state_dict(torch.load(pre_trained_weight, map_location=device))

unet.to(device)
model.to(device)

# data
dir_ = '../data/tanjania/raw_data/'
# image_masks = glob.glob(os.path.join(*[dir_, 'unet_binary', '*.tif']))
images = glob.glob(os.path.join(*[dir_, 'test', '*.tif']))
geojsons = glob.glob(os.path.join(*[dir_, 'test', '*.geojson']))
# image_masks.sort()
images.sort()
geojsons.sort()
assert len(images) == len(geojsons) != 0

image_id = 0
start_ann_id = 1
gt_annotation_dicts = []
dt_annotation_dicts = []
image_dicts = []
gt_gdfs, dt_gdfs = [], []

# loop through images and get image masks
for image_, geojsons_ in zip(images, geojsons):
    # read RGB image
    print(f'Processing ', os.path.basename(image_), os.path.basename(geojsons_))
    image_filename = os.path.basename(image_)
    out_file = os.path.join(*[dir_, 'dinov2cls', image_filename])
    # assert mask_filename == os.path.basename(image_)
    out_rows = []

    # get binary mask and probability mask  # fixme write parallel processing and split big image into samll image
    pred_size = 1024
    patch_size = pred_size * 7
    stride = patch_size - 256
    output = predict_image(image_, unet, device=device, batch_size=1, patch_size=patch_size, stride=stride)
    class_array = np.argmax(output, axis=0)
    score_array = np.amax(output, axis=0)

    # convert class_array to gdf of polygons and add unet confidence score
    # import ipdb; ipdb.set_trace()
    # dt_gdf_test = vectorise(class_array, image_, score_array=score_array)

    # create polygons # fixme create separate function and do parallell processing
    with rasterio.open(image_) as src_:
        cls_meta = src_.meta.copy()
        cls_meta.update(dtype='uint8')
        cls_meta.update(count=1)

        with rasterio.io.MemoryFile() as cls_memfile:
            with cls_memfile.open(**cls_meta) as cls_dst:
                cls_dst.write(class_array, 1)
            with cls_memfile.open() as cls_dataset:
                mask = cls_dataset.read(1)

                # read probability mask
                prob_meta = cls_meta.copy()
                prob_meta.update(dtype=score_array.dtype)
                with rasterio.io.MemoryFile() as prob_memfile:
                    with prob_memfile.open(**prob_meta) as prob_dst:
                        prob_dst.write(score_array, 1)
                    with prob_memfile.open() as prob_dataset:
                        prob = prob_dataset.read(1)

                        # iterate through objects in mask and generate input data with 448x448 patches
                        mask = np.where(mask == 0, 0, 1)
                        final_image = np.zeros_like(mask)
                        if np.sum(mask) != 0:
                            labeled_image, objects_count = ndi.label(mask)
                            print("Object count: ", objects_count)
                            bounding_boxes = ndi.find_objects(labeled_image)
                            windows = [bbox_to_window(bbox) for bbox in bounding_boxes]

                            # Create a new objects image in memory
                            obj_meta = cls_meta.copy()
                            obj_meta.update(dtype=labeled_image.dtype)
                            with rasterio.io.MemoryFile() as obj_memfile:
                                with obj_memfile.open(**obj_meta) as dst:
                                    dst.write(labeled_image, 1)
                                with obj_memfile.open() as dataset:
                                    for object_id, door in tqdm(enumerate(windows)):
                                        object_image = np.where(labeled_image == object_id + 1, 1, 0).astype('uint8')
                                        if np.sum(object_image) > 100:
                                            # get features from DINOv2
                                            mask_arr = dataset.read(1, window=door, boundless=True, fill_value=0)
                                            mask_arr = np.where(mask_arr == object_id + 1, 1, 0).astype('uint8')
                                            image_arr = src_.read((1, 2, 3), window=door, boundless=True, fill_value=0)
                                            X = torch.unsqueeze(torch.tensor(image_arr / 255), 0).float().to(device)
                                            m = torch.unsqueeze(torch.tensor(mask_arr), 0).long().to(device)
                                            pred = model(X, m)
                                            features = pred.logits
                                            features = features.squeeze().cpu().detach().numpy().reshape(1, -1)
                                            # predict using SVM
                                            y_prob_test = svm.predict_proba(features)
                                            y_pred_test = np.argmax(y_prob_test, axis=1)[0] + 1
                                            final_image = np.where(labeled_image == object_id + 1, y_pred_test,
                                                                   final_image)

                                            # get prob score from unet
                                            prob_arr = prob_dataset.read(1, window=door, boundless=True, fill_value=0)
                                            prob_arr = np.where(mask_arr == 1, prob_arr, 0)
                                            unet_score = np.sum(prob_arr) / np.sum(mask_arr)

                                            # write annotaion after converting into polygon
                                            score_ = np.amax(y_prob_test, axis=1)[0]
                                            results = list(shapes(object_image, transform=dataset.transform))
                                            polygon = [shape(geom) for geom, value in results if value == 1]
                                            rows = [image_id, score_, y_pred_test, unet_score, polygon]
                                            out_rows.append(rows)
                                        else:
                                            # write 0 in small objext places
                                            print(np.sum(object_image))
                                            final_image = np.where(labeled_image == object_id + 1, 0, final_image)
                            # write class image
                            with rasterio.open(out_file, 'w', **cls_meta) as dst:
                                dst.write(final_image, 1)
                        else:
                            #  write blank image
                            with rasterio.open(out_file, 'w', **cls_meta) as dst:
                                dst.write(mask, 1)

    # image dictionary for coco format
    image_dict = {"file_name": image_, "height": src_.meta['height'], "width": src_.meta['width'],
                  "id": image_id}
    image_dicts.append(image_dict)
    gt_gdf = gpd.read_file(geojsons_)
    rows__, _ = gt_gdf.shape
    patch_trans = get_gdal_transform(image_)
    if rows__ > 0:
        # import ipdb; ipdb.set_trace()
        ann_dict_gt, start_ann_id = coco_annotation_dict(gt_gdf, classifier_name, image_id,
                                                         patch_trans, start_ann_id)
        gt_annotation_dicts += ann_dict_gt
        gt_gdfs.append(gt_gdf)

    attributes_list = [elem[:-1] for elem in out_rows]  # all elements except the last
    geometry_list = [elem[-1][0] for elem in out_rows]
    dt_df = pd.DataFrame(attributes_list, columns=['image_id', 'score', 'label', 'unet_score'])
    dt_gdf = gpd.GeoDataFrame(dt_df, geometry=geometry_list)
    if dt_gdf.shape[0] > 0:
        ann_dict_dt, _ = coco_annotation_dict(dt_gdf, 'label', image_id,
                                              patch_trans, ann_type='dt', extra_attr='unet_score')
        dt_annotation_dicts += ann_dict_dt
        dt_gdfs.append(dt_gdf)

    image_id += 1
    # if image_id == 2:
    #     break

# write coco files
coco_filename = f'annotation_dinov2cls_test_{classifier_name}.json'
outfile_name = os.path.join(dir_, coco_filename)
print(f"Number of objects are: {len(dt_annotation_dicts)}")
with open(outfile_name, 'w') as f:
    json.dump(dt_annotation_dicts, f)

# # generate ground truth cosco json
# coco_json_file = os.path.join(*[dir_, 'data', 'annotations.json'])
final_coco_dict = custom_coco_from_polygons(image_dicts, gt_annotation_dicts, dataset_name='nacala')
coco_filename = f'annotation_test_{classifier_name}.json'
save_coco_json(os.path.join(dir_, coco_filename), final_coco_dict)

# write shefiles
dt_final_gdf = gpd.GeoDataFrame(pd.concat(dt_gdfs, ignore_index=True))
dt_final_gdf.to_file(os.path.join(dir_, f'dt_data_{classifier_name}.shp'))
gt_final_gdf = gpd.GeoDataFrame(pd.concat(gt_gdfs, ignore_index=True))
gt_final_gdf.to_file(os.path.join(dir_, 'gt_data.shp'))
