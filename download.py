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
    "unet1", "unet2", "unet3", "unet4", "unet5",
    "unet_m1", "unet_m2", "unet_m3", "unet_m4", "unet_m5",
    "unet_e1", "unet_e2", "unet_e3", "unet_e4", "unet_e5",
    "unet_2h1", "unet_2h2", "unet_2h3", "unet_2h4", "unet_2h5",
    "unet_2h_m1", "unet_2h_m2", "unet_2h_m3", "unet_2h_m4", "unet_2h_m5",
    "unet_2d1", "unet_2d2", "unet_2d3", "unet_2d4", "unet_2d5",
    "yolo1", "yolo2", "yolo3", "yolo4", "yolo5",
    "yolo_m1", "yolo_m2", "yolo_m3", "yolo_m4", "yolo_m5",
    "dino_m1", "dino_m2", "dino_m3", "dino_m4", "dino_m5"
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

