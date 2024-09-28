# Nacala
## Nacala-Roof-Material: Drone Imagery for Roof Classification, Segmentation, and Counting to Support Mosquito-borne Disease Risk Assessment

## Data
All the data and trained models used in the paper are provided in this zipfile.
* The training, validation and test sets are provided in  `train.zip` and `test.zip` files. The `train.zip` file contains both `train` and `validation` sets.
* The second test set (external test set) provided in `test2.zip` file.
* The trained models are also provided and can be accessed from `models` folder.
* The DINOv2 features are provided in `dinov2_features.zip` folder

* The Nacala-Roof-Material data are provided in the folder structure as below.
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
* The sample datasets are provided in the `sample` folder, that is placed inside `code/datasets` folder.
* The sample can be used with all the scripts provided in the repository.
* The Nacala-Roof-Material dataset should be placed inside `datasets` folder at the same level as sample dataset.
* All files in `train`, `valid` and `test` are in `*.tif` format except for YOLOv8 labels in `*.txt` format.
* The training Python file for YOLOv8 accesses the required data in `*.yaml` format.
* The images and labels for the test set are provided in `*.tif` and `*.geojson` format, respectively.
* The evaluation script takes these files and returns all metrics.
Map data sourced from [OpenStreetMap](https://www.openstreetmap.org/copyright).

## Prepare data before using the code
* Before running the scripts, the Nacala-Roof-Material data (`train.zip`, `test.zip`, `test2.zip`) should be placed in the `code/datasets/nacala` folder and extract them.
* Place `models` folder also inside `datasets` folder.

## Code
All python scripts provided in code folder, for training and testing the models considered in the correspoding resaerch paper.
Example are provided for testing and training these models.

## Train and Evaluate Models

### Installations
Create conda environment and install required packages. The provided package versions worked well in our device. <br>
`conda create -n nacala python=3.10` <br>
`conda activate nacala` <br>
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` <br>
`cd code` <br>
`pip install -r requirements.txt` <br>

### Evaluate Models
Once the test set placed and extracted ino the `datasets` folder, perfromance metrics can be obtaied on the test set.

Two testing scripts provided in the repository. `pred_seg.py` for all segmentation models and `pred_yolov8.py` for YOLOv8.

#### Examples
* Evaluate UNet: <br />
```python pred_seg.py --patch_size 4800 --use_dinov2cls --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/unet1/```

* Evaluate UNet<sub>Multi</sub>: <br>
```python pred_seg.py --patch_size 4800 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/unet_multi1/ --num_classes 6``` <br>

* Evaluate UNet<sub>DOW</sub>: <br>
```python pred_seg.py --model_name unet_2heads --patch_size 4800 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/unet_dow1/ --use_dinov2cls``` <br>

* Evaluate UNet<sub>DOW-Multi</sub>: <br>
```python pred_seg.py --model_name unet_2heads --patch_size 4800 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/unet_dow_multi1/ --loss_type cross_entropy_cls3 --num_classes 6``` <br>

* Evaluate DINOv2: <br>
```python pred_seg.py --model_name dinov2 --patch_size 512 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/dinov21/ --use_dinov2cls``` <br>

* Evaluate DINIv2<sub>Multi</sub>: <br>
```python pred_seg.py --model_name dinov2 --patch_size 512 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/dinov2_multi1/ --num_classes 6``` <br>

* Evaluate DINOv2<sub>DOW</sub>: <br>
```python pred_seg.py --model_name dinov2 --patch_size 512 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/dinov2_dow1/ --num_classes 1 --use_dinov2cls``` <br>

* Evaluate DINOv2<sub>DOW-Multi</sub>: <br>
```python pred_seg.py --model_name dinov2_2heads --patch_size 512 --data_dir ./datasets/nacala/test/ --weights_folder ./datasets/models/dinov2_dow_multi1/ --loss_type cross_entropy_cls3 --num_classes 6``` <br>

* Evaluate YOLOv8<sub>Multi</sub>: <br>
```python pred_yolov8.py --data_dir ./datasets/test/ --weights_folder ./datasets/nacala/models/yolo_multi1/``` <br>

* Evaluate YOLOv8: <br>
```python pred_yolov8.py --data_dir ./datasets/test/ --weights_folder ./datasets/nacala/models/yolo1/ --use_dinov2cls``` <br>


### Training models
`train_seg.py` is for training the simple UNet/DINOv2 and `train_dow.py` is used to train the UNet/DINOv2 with DOW.

Examples:
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

### Notes
1. To train YOLOv8 for binary labels of sample data, the folder with images has to be copied to the `datasets/sample/yolo_binary` directory.
Same for the Nacala-Roof-Material or a custom dataset.
2. For reference, all file names that can be downloaded can find in the `download.py` file.
3. The default arguments for evaluation and training scripts can be found in the respective Python files (e.g., `pred_seg.py`, `train_seg.py`).
4. Please open the issue if you encounter any problems with the code or data.
