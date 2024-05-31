import os
import sys
import os
sys.path.append('../')
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, help="Model name", default="test")
    parser.add_argument("--data_path", type=str, help="Model name",
                        default='./data/sample/train/data.yaml')
    args = parser.parse_args()
    keyword = args.keyword

    # Load a model
    model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
    print(os.getcwd())
    # Classification model with hyperparameters tuned on train dataset
    model.train(data=args.data_path,
                epochs=500,
                val=True,
                batch=128,  # batch size
                label_smoothing=0.1,
                close_mosaic=0,  # 0 for closing mosaic
                optimizer='AdamW',  # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                patience=0,  # early stopping
                name=keyword,
                amp=False,
                lr0=0.00412,
                lrf=0.00816,
                momentum=0.85315,
                weight_decay=0.0006,
                warmup_epochs=4.88943,
                warmup_momentum=0.78702,
                box=3.92059,
                cls=0.2171,
                dfl=1.47369,
                hsv_h=0.01421,
                hsv_s=0.38183,
                hsv_v=0.41388,
                degrees=0.0,
                translate=0.06837,
                scale=0.27428,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.3917,
                mosaic=0.87783,
                mixup=0.0,
                copy_paste=0.0,
                )  # unet the model
    metrics = model.val()  # evaluate model performance on the validation set
    print(metrics)
