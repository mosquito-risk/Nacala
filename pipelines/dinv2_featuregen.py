"""
Code for generating DINOv2 features using batches of images and mask.
This code should be used with cnn_patch_gen.py code to generate features for each patch.
"""

import os, sys
import torch
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.models import dinov2_cls
from pipelines import base_class


class DINOv2FeatureGen(base_class.BaseClass):
    def __init__(self, image_array, mask_array, label_array=None, num_classes=5, batch_size=2):
        super().__init__()
        self.image_array = image_array
        self.mask_array = mask_array
        self.label_array = label_array
        self.features_array = None
        self.num_classes = num_classes
        self.device = self.get_device()
        self.batch_size = batch_size

    def load_model(self):
        id2label = list(range(self.num_classes))
        model = dinov2_cls.Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",
                                                                         id2label=id2label,
                                                                         num_labels=self.num_classes,
                                                                         num_classes=self.num_classes,
                                                                         avg_pool=True)
        return model

    # load data and get features
    def get_features(self):
        model = self.load_model()
        model.to(self.device)

        for i in tqdm(range(0, len(self.image_array), self.batch_size)):
            X = torch.tensor(self.image_array[i:i + self.batch_size]).float().to(self.device)
            X /= 255
            m = torch.tensor(np.expand_dims(self.mask_array[i:i + self.batch_size], axis=1)).long().to(self.device)
            pred = model(X, m)
            features = pred.logits
            features = features.squeeze().cpu().detach().numpy()
            if len(features.shape) == 1:
                features = np.expand_dims(features, axis=0)
            if self.label_array is not None:
                extra_cols = self.label_array[i:i + self.batch_size]
                features = np.hstack([features, extra_cols])
            if self.features_array is None:
                self.features_array = features
            else:
                self.features_array = np.vstack([self.features_array, features])
        return self.features_array
