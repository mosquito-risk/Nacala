"""
Code for
1. Segmenting image using FCN
2. Preicting label to each segment using CNN/DINOv2-cls models
3. Saving the results as a shapefile or image
4. Get score from different sources (e.g. UNet's Softmax later, DINOv2-cls (from classifier), etc.)
5. Save COCO format results
"""

import json
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pipelines import fcn_prediction
from utils.different_labels import watershed_image


class FCNCNNPredict(fcn_prediction.FCNPredict):
    def __init__(self, image_dir, weights_path, model_name, num_classes, patch_size, stride,
                 cnn_wights_path=None, classifier_path=None, keyword='test', out_dir='./', pred_batch_size=1,
                 energy_levels=False, mask_decision="both", head_size='n', dt_geojson=None, loss_type=None,
                 use_dinov2cls=False):
        super().__init__(image_dir=image_dir, weights_path=weights_path, model_name=model_name, num_classes=num_classes,
                         patch_size=patch_size, stride=stride, keyword=keyword, classifier_path=classifier_path,
                         out_dir=out_dir, energy_levels=energy_levels, mask_decision=mask_decision, head_size=head_size,
                         dt_geojson=dt_geojson, use_dinov2cls=use_dinov2cls)
        self.cnn_model_path = cnn_wights_path
        self.image_list = self.base_class.create_image_list()
        self.label_list = self.base_class.create_label_list()
        self.out_dir = out_dir
        self.pred_batch_size = pred_batch_size
        self.loss_type = loss_type
        self.score_attr = 'score'

    def predict_all_images(self):
        for idx, image_path in enumerate(self.image_list):
            # predict using multi head UNet models
            if self.model_name == 'unet_2heads' or self.model_name == 'unet_2decoders':
                class_array, score_array = self.predict_multi_output(image_path, mask_decision=self.mask_decision)
                self.binary_label = True
            elif self.energy_levels:
                output = self.predict_image(image_path, loss_type=self.loss_type)
                score_array = np.amax(output, axis=0)
                energy_map = (output >= 0.5).cumprod(0).sum(0)
                class_array = watershed_image(energy_map)
                self.binary_label = True
            else:
                output = self.predict_image(image_path, loss_type=self.loss_type)
                if self.num_classes == 1:
                    class_array = (output >= 0.5).astype(np.uint8)
                    score_array = output.squeeze(0)
                else:
                    class_array = np.argmax(output, axis=0).astype(np.uint8)
                    score_array = np.amax(output, axis=0)

            # create geodata frame with polygons
            meta = self.output_metadata(image_path)
            polys = self.vectorise_image(class_array, meta, binary_label=self.binary_label)
            print(f'Number of segments in the image: {len(polys)}')

            if self.label_list:
                label_fp = self.label_list[idx]
                label_gdf = gpd.read_file(label_fp)
                print(f'Number of ground truth polygons: {len(label_gdf)}')

            # load classifier and predict
            if len(polys) == 0:
                continue
            # add unet score to polygons
            polys['score'] = self.add_fcn_score_to_polys_v2(polys, score_array, meta)
            print(f'Added FCN Score to polygons')

            # create batch for DINOv2 model input
            if self.use_dinov2cls:
                polys = self.add_score_class_to_polys(image_path, polys)
                print(f'Added DINOv2 Score to polygons')

            # self.coco_annotation += ann_dict_dt
            self.dataframes.append(polys)
            self.image_id += 1

        # save dataframes
        dt_final_gdf = gpd.GeoDataFrame(pd.concat(self.dataframes, ignore_index=True))
        dt_final_gdf.to_file(self.dt_geojson)
