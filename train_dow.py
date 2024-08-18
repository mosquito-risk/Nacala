"""
Training UNet-2heads and UNet-2decoders
"""

import os
import sys
import json
import argparse

sys.path.append("../..")
from pipelines.training import training_2heads, training_2heads_v2

# model details, training details, data details
params = {
    'encoder_weights': 'imagenet',
    'model_weights': None,
    'batch_size_val': 12,
    'optimizer': 'adamw',
    'learning_rate': 0.0003,
    'patch_size': 512,
    'image_folder': "images",
    'label1_folder': "p_labels",
    'weight_folder': 'weights',
    'train_folder': "train"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str,
                        help="Keyword is unique name for saving trained model and tensorbooard logs", default="test")
    parser.add_argument("--model_name", type=str,
                        help="Name of the model from pipelines/models/load_models.py", default="unet_2heads")
    parser.add_argument("--label2_folder", type=str, help="Second head labels", default="int_mask")
    parser.add_argument("--train_batch_size", type=int, help="Batch size of training data", default=16)
    parser.add_argument("--sample_size", type=int,
                        help="sample size of training and validation if not using whole dataset", default=None)
    parser.add_argument("--num_classes", type=int, help="Number of classes for each head", default=1)
    parser.add_argument("--load_data2device", type=str, help="device to load data CPU/GPU memory",
                        default="cuda")
    parser.add_argument("--loss_type", type=str,
                        help="type of loss function (see loss functions in pipelines/training/loss_functions.py",
                        default="cross_entropy")
    parser.add_argument("--head_size", type=str, help="Size of head for 2 head model", default="n")
    parser.add_argument("--tensordata_folder", type=str,
                        help="Save whole data as a sinlge file in this directory", default="./input")
    parser.add_argument("--load_data2memory", action=argparse.BooleanOptionalAction,
                        help="Whether to load data from disk to CPU/GPU memory", default=False)
    parser.add_argument("--t_params", nargs=3, help="Tversky loss parameters (Alpha, Beta and Gamma)",
                        type=float, default=(0.7, 0.3, 1.0))
    parser.add_argument("--out_dir", type=str, help="Output directory to save trained model and logs",
                        default="./temp")
    parser.add_argument("--data_path", type=str, help="Path to data folders",
                        default='./data/sample/')
    parser.add_argument("--val_folder", type=str, help="Validation folder name", default="valid")
    parser.add_argument("--epochs", type=int, help="Training epochs", default=300)
    args = parser.parse_args()

    keyword = args.keyword
    params['keyword'] = keyword
    params['num_classes'] = args.num_classes
    params['model_name'] = args.model_name
    params['batch_size_train'] = args.train_batch_size
    params['label2_folder'] = args.label2_folder
    params['load_data2memory'] = args.load_data2memory
    params['load_data2device'] = args.load_data2device
    params['sample_size'] = args.sample_size
    params['loss_type'] = args.loss_type
    params['head_size'] = args.head_size
    params['t_params'] = args.t_params
    params['epochs'] = args.epochs
    # params['out_dir'] = args.out_dir
    params['out_dir'] = "/scratch/project_465001005/projects/nacala/rebuttal_dir/output"
    if os.path.exists(params['out_dir']) is False:
        os.makedirs(params['out_dir'])
    params['log_dir'] = os.path.join(args.out_dir, "logs")
    # params['data_path'] = args.data_path
    params['data_path'] = "/scratch/project_465001005/projects/nacala/rebuttal_dir/data/"
    params['tensordata_folder'] = os.path.join(params['data_path'], args.tensordata_folder)
    params['val_folder'] = args.val_folder

    # print params
    print("Parameters: \n", json.dumps(params, indent=4))
    # save params as json
    with open(f'{params["out_dir"]}/train_{keyword}.json', 'w') as file:
        json.dump(params, file, indent=4)

    # initialize training object and train model
    if args.loss_type == 'cross_entropy_cls':
        train = training_2heads_v2.TrainSegmentation(**params)
    else:
        train = training_2heads.TrainSegmentation(**params)
    train.train_model()
