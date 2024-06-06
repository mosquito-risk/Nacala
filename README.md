# Nacala
## Nacala-Roof-Material: Drone Imagery for Roof Classification, Segmentation, and Counting to Support Mosquito-borne Disease Risk Assessment

The project homepage can be found here: [https://mosquito-risk.github.io/Nacala](https://mosquito-risk.github.io/Nacala/)

## Data
All the data and trained models used in the correspoding paper can be accessed through this link: [Data and Models](https://sid.erda.dk/sharelink/aHw1Pey5BC).
* The training, validation and test sets are provided in `test.zip` and `train.zip` files. The `train.zip` file contains both `train` and `valid` sets.
* The trained models are also provided and can be accessed in the `models` folder in the same link
* The data and models can be dowloaded directly by going to the link or using `download.py` file
* The below command is an example for downloading and extracting the test set (for train set `--filename` is `train`) <br />
  ```python download.py --filename test --outdir ./datasets``` <br />


### Folder structure
```
datasets                             # Data folder
└───sample                           # Sample dataset
    └───train
    │   └───energy_mask              # Six energy level mask for UNet_DOW-6
    │   └───images                   # RGB Images
    │   └───int_mask                 # Only interior mask for UNet_DOW
    │   └───labels                   # Labels for YOLOv8 in *.txt
    │   └───p_labels                 # Patch labels for all segmentation models
    │   └───weights                  # Weight mask for all segmentation model
    └───valid
    │   └───...(same as train)
    └───test
          0000000000.tif
          0000000000.geojson
          ...
└───custom_dataset 
    └───(same as sample)
```
The Nacala-Roof-Material data are provided in the same folder structure as above.
The data are provided as patches as well as raw data. The patches with all labels can be generated
with the `patch_gen.py` script.
The sample dataset provided in the `datasets/sample` folder.
The sample can be used with all the scripts provided in the repository.
The Nacala-Roof-Material dataset should be placed in the datasets folder at the same level as sample.
All files in `train`, `valid` and `test` are in `*.tif` format except for YOLOv8 labels in `*.txt` format.
The training Python file for YOLOv8 accesses the required data in `*.yaml` format.
The images and labels for the test set are provided in `*.tif` and `*.geojson` format, respectively.
The evaluation script takes these files and returns all metrics.

Map data sourced from [OpenStreetMap](https://www.openstreetmap.org/copyright).

## Code
This repository contains the code for training and testing the models considered in the correspoding resaerch paper.
Example are provided for testing and training these models.

## Train and Evaluate Models

### Installations
Create conda environment and install required packages. The provided package versions worked well in our device. <br>
`conda create -n nacala python=3.10` <br>
`conda activate nacala` <br>
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` <br>
`pip install -r requirements.txt` <br>

### Evaluate Models
Once the test set is downloaded and extracted to the `datasets` folder,
the pretrained models can be downloaded using same command, and perfromance metrics can be obtaied on the test set.
Two testing scripts provided in the repository. `pred_seg.py` for all segmentation models
and `pred_yolov8.py` for YOLOv8.
All trained models can be accessed via [trained models](https://sid.erda.dk/sharelink/HF2srDrYEa).

* Evaluate UNet: <br />
To download UNet and classifier weights, use the following commands.
The classifier is logistic regression. <br>
```python download.py --filename unet1 --outdir ./temp/``` <br>
```python download.py --filename classifier --outdir ./temp/``` <br>
```python pred_seg.py --patch_size 4800 --use_dinov2cls --data_dir ./datasets/test/ --weights_folder ./temp/unet1/```
* Evaluate UNet<sub>Multi</sub>: <br>
```python download.py --filename unet_multi1 --outdir ./temp/``` <br>
```python pred_seg.py --patch_size 4800 --data_dir ./datasets/test/ --weights_folder ./temp/unet_multi1/ --num_classes 6``` <br>

[//]: # (* Evaluate UNet<sub>DOW-6</sub>: <br>)
[//]: # (```python download.py --filename unet_e1 --outdir ./temp/``` <br>)
[//]: # (```python pred_segpy --patch_size 4800 --data_dir ./datasets/test/ --weights_folder ./temp/unet_e1/ --num_classes 6 --use_dinov2cls --energy_levels --loss_type ordinal``` <br>)

* Evaluate UNet<sub>DOW</sub>: <br>
```python download.py --filename unet_dow1 --outdir ./temp/``` <br>
```python pred_seg.py --model_name unet_2heads --patch_size 4800 --data_dir ./datasets/test/ --weights_folder ./temp/unet_dow1/ --use_dinov2cls``` <br>
* Evaluate UNet<sub>DOW-Multi</sub>: <br>
```python download.py --filename unet_dow_multi1 --outdir ./temp/``` <br>
```python pred_seg.py --model_name unet_2heads --patch_size 4800 --data_dir ./datasets/test/ --weights_folder ./temp/unet_dow_multi1/ --loss_type cross_entropy_cls``` <br>
* Evaluate DINOv2: <br>
```python download.py --filename dinov21 --outdir ./temp/``` <br>
```python pred_seg.py --model_name dinov2 --patch_size 4800 --data_dir ./datasets/test/ --weights_folder ./temp/dinov21/ --num_classes 6``` <br>
* Evaluate YOLOv8<sub>Multi</sub>: <br>
```python download.py --filename yolo_multi1 --outdir ./temp/``` <br>
```python pred_yolov8.py --data_dir ./datasets/test/ --weights_folder ./temp/yolo_multi1/``` <br>
* Evaluate YOLOv8: <br>
```python download.py --filename yolo1 --outdir ./temp/``` <br>
```python pred_yolov8.py --data_dir ./datasets/test/ --weights_folder ./temp/yolo1/ --use_dinov2cls``` <br>

[//]: # (* Evaluate UNet<sub>2decoders</sub>: <br>)
[//]: # (```python download.py --filename unet_2d1 --outdir ./temp/``` <br>)
[//]: # (```python pred_unet.py --model_name unet_2decoders --patch_size 4800 --data_dir ./datasets/test/ --weights_folder ./temp/unet_2d1/ --use_dinov2cls``` <br>)


### Training models
`train_unet.py` is for training the simple UNet and `train_unet_2heads.py` is used to train the UNet with DOW.
For example:
* Training simple UNet: <br />
```python train_seg.py --keyword test1 --val_folder train --out_dir temp --data_path ./datasets/sample/ --use_border_weight```
* Training UNet<sub>Multi</sub>: <br />
```python train_seg.py --keyword test2 --val_folder train --num_classes 3 --out_dir temp --data_path ./datasets/sample/ --use_border_weight```
* Training UNet<sub>DOW-6</sub>: <br />
```python train_seg.py --keyword test3 --val_folder train --num_classes 6 --out_dir temp --data_path ./datasets/sample/ --label_folder energy_mask --loss_type ordinal_cross_entropy```
* Training UNet<sub>DOW</sub>: <br />
```python train_dow.py --keyword test4 --val_folder train --label2_folder int_mask --out_dir temp --data_path ./datasets/sample/```
* Training UNet<sub>DOW-Multi</sub>: <br />
```python train_dow.py --keyword test6 --model_name unet_2heads --val_folder train --label2_folder int_mask --out_dir temp --data_path ./datasets/sample/ --loss_type cross_entropy_cls```
* Training DINOv2: <br />
```python train_seg.py --keyword test7 --model_name dinov2 --val_folder train --out_dir temp --data_path ./datasets/sample/ --num_classes 3```
* Training YOLOv8<sub>Multi</sub>: <br />
```python train_yolov8.py --keyword test8 --data_path ./datasets/sample/train/data.yaml```
* Training YOLOv8: <br />
```python train_yolov8.py --keyword test9 --data_path ./datasets/sample/yolo_binary/data.yaml```

[//]: # (* Training UNet<sub>2decoders</sub>: <br />)
[//]: # (```python train_unet_2heads.py --keyword test5 --model_name unet_2decoders --val_folder train --label2_folder int_mask --out_dir temp --data_path ./datasets/sample/```)

### Notes
1. To train YOLOv8 for binary labels of sample data, the folder with images has to be copied to the `datasets/sample/yolo_binary` directory.
Same for the Nacala-Roof-Material or a custom dataset.
2. For reference, all file names that can be downloaded can find in the `download.py` file.
3. The default arguments for evaluation and training scripts can be found in the respective Python files (e.g., `pred_seg.py`, `train_seg.py`).
4. Please open the issue if you encounter any problems with the code or data.
