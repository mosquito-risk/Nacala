import sys

sys.path.append('../')
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, help="Model name", default="test")
    parser.add_argument("--data_path", type=str, help="Model name",
                        default='../../sample_data/ready_seg/train/data.yaml')
    args = parser.parse_args()
    keyword = args.keyword

    # Load a model
    model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n-seg.pt")  # load a model from a file

    # Classification model with hyperparameters tuned on train dataset
    model.train(data=args.data_path,
                imgsz=640,
                epochs=100,
                val=True,
                batch=4,  # batch size
                label_smoothing=0.1,
                name=keyword,
                amp=False,
                close_mosaic=0,  # 0 for closing mosaic
                optimizer='AdamW',  # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                patience=0,  # early stopping
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
                scale=0,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.3917,
                # bgr=0.0,
                mosaic=0,
                mixup=0.0,
                copy_paste=0.0,
                save_json=True,
                mask_ratio=1,
                overlap_mask=False
                )  # unet the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # print(metrics)