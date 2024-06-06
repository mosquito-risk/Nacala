"""
Download and extract the data
List of available datasets:
- train: train and validation sets
- test: test set
- trained_models: all trained models
- raw_data: The raw data of train and validation sets
- dinov2_features: The DINOv2 features extracted from from all buildings: train, validation, and test sets
"""

import os
import argparse
from torchvision.datasets.utils import download_url, extract_archive

# List of trained models
trained_models = [
    "classifier",
    "yolo1", "yolo2", "yolo3", "yolo4", "yolo5",                                                          # YOLOv8
    "dinov21", "dinov22", "dinov23", "dinov24", "dinov25",                                                # DINOv2
    "dinov2_dow1", "dinov2_dow2", "dinov2_dow3", "dinov2_dow4", "dinov2_dow5",                            # DINOv2_DOW
    "unet1", "unet2", "unet3", "unet4", "unet5",                                                          # UNet
    "unet_dow1", "unet_dow2", "unet_dow3", "unet_dow4", "unet_dow5",                                      # UNet_DOW
    "yolo_multi1", "yolo_multi2", "yolo_multi3", "yolo_multi4", "yolo_multi5",                            # YOLO_Multi
    "dinov2_multi1", "dinov2_multi2", "dinov2_multi3", "dinov2_multi4", "dinov2_multi5",                  # DINOv2_Multi
    "dinov2_dow_multi1", "dinov2_dow_multi2", "dinov2_dow_multi3", "dinov2_dow_multi4", "dinov2_dow_multi5",
    "unet_multi1", "unet_multi2", "unet_multi3", "unet_multi4", "unet_multi5",                            # UNet_Multi
    "unet_dow_multi1", "unet_dow_multi2", "unet_dow_multi3", "unet_dow_multi4", "unet_dow_multi5",     # UNet_DOW_Multi
]

# List of available datasets
datasets = ["train", "test", "raw_data", "dinov2_features"]


def download_data(dataset_name: str, outdir: str):
    """ Download the data """
    # check filename and define the URL
    filename = f"{dataset_name}.zip"
    if dataset_name in datasets:
        url = f"https://sid.erda.dk/share_redirect/fOxSHwH5hr/{filename}"
    elif dataset_name in trained_models:
        url = f"https://sid.erda.dk/share_redirect/HF2srDrYEa/{filename}"
    else:
        raise ValueError(f"Dataset/ Model name: {dataset_name} is not available.")

    # Download and extract file.
    download_url(url, ".", filename)
    print(f"Downloaded {filename}")
    extract_archive(filename, outdir)
    print(f"Extracted {filename} to {outdir}")
    os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="filename to download", default="lumi")
    parser.add_argument("--outdir", type=str, help="Out directory to extract the zip file", default="./temp")
    args = parser.parse_args()
    download_data(args.filename, args.outdir)

