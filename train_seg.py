"""
Training UNet model for binary, mutli-class segmentation and also ordinal segmentation.
Example: python train_unet.py --keyword test_binary --model_name uresnet34 --val_folder train --train_batch_size 4
 --num_classes 1 --epochs 10 --out_dir temp --data_path ./data/sample/
"""

import os
import sys
import json
import argparse

sys.path.append("../..")
from pipelines.training import training

# model, training and data details, can be modified here or using command line arguments
params = {
    'encoder_weights': 'imagenet',
    'model_weights': None,
    'batch_size_val': 16,
    'epochs': 300,
    'optimizer': 'adamw',
    'learning_rate': 0.0003,
    'patch_size': 512,
    'image_folder': "images",
    'weight_folder': 'weights',
    'train_folder': "train",
    'border_threshold': 7
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str,
                        help="Keyword is unique name for saving trained model and tensorbooard logs", default="test")
    parser.add_argument("--model_name", type=str,
                        help="Name of the model from pipelines/models/load_models.py", default="uresnet34")
    parser.add_argument("--label_folder", type=str, help="label folder for training", default="p_labels")
    parser.add_argument("--load_data2memory", action=argparse.BooleanOptionalAction,
                        help="Whether to load data from disk to CPU/GPU memory", default=False)
    parser.add_argument("--load_data2device", type=str, help="device to load data CPU/GPU memory",
                        default="cuda")
    parser.add_argument("--tensordata_folder", type=str,
                        help="Save whole data as a sinlge file in this directory", default="./input")
    parser.add_argument("--loss_type", type=str,
                        help="type of loss function (see loss functions in pipelines/training/loss_functions.py",
                        default="cross_entropy")
    parser.add_argument("--use_border_weight", action=argparse.BooleanOptionalAction,
                        help="Use border weight separate building those are clouse using weight mask", default=False)
    parser.add_argument("--train_batch_size", type=int, help="Batch size of training data", default=16)
    parser.add_argument("--num_classes", type=int, help="Number of classes to train", default=1)
    parser.add_argument("--sample_size", type=int,
                        help="sample size of training and validation if not using whole dataset", default=None)
    parser.add_argument("--epochs", type=int, help="Training epochs", default=300)
    parser.add_argument("-cw", nargs=6, metavar=('cbg', 'c1', 'c2', 'c3', 'c4', 'c5'),
                        help="Background and class weights", type=float, default=None)
    parser.add_argument("--t_params", nargs=3, help="Tversky loss parameters (Alpha, Beta and Gamma)",
                        type=float, default=(0.7, 0.3, 1.0))
    parser.add_argument("--out_dir", type=str, help="Output directory to save trained model and logs",
                        default="./temp")
    parser.add_argument("--data_path", type=str, help="Path to data folders file",
                        default='./data/sample/')
    parser.add_argument("--val_folder", type=str, help="Validation folder name", default="valid")
    args = parser.parse_args()

    keyword = args.keyword
    params['keyword'] = keyword
    params['use_border_weight'] = args.use_border_weight
    params['model_name'] = args.model_name
    params['batch_size_train'] = args.train_batch_size
    params['num_classes'] = args.num_classes
    params['label_folder'] = args.label_folder
    params['sample_size'] = args.sample_size
    params['load_data2memory'] = args.load_data2memory
    params['load_data2device'] = args.load_data2device
    params['epochs'] = args.epochs
    params['loss_type'] = args.loss_type
    params['t_params'] = args.t_params
    # params['out_dir'] = args.out_dir
    params['out_dir'] = "/scratch/project_465001005/projects/nacala/rebuttal_dir/output"
    if os.path.exists(params['out_dir']) is False:
        os.makedirs(params['out_dir'])
    params['log_dir'] = os.path.join(args.out_dir, "logs")
    # params['data_path'] = args.data_path
    params['data_path'] = "/scratch/project_465001005/projects/nacala/rebuttal_dir/data/"
    params['tensordata_folder'] = os.path.join(params['data_path'], args.tensordata_folder)
    params['val_folder'] = args.val_folder

    # class weights in dictionary format
    if args.cw is not None:
        params['weights_dict'] = {1: args.cw[1], 2: args.cw[2], 3: args.cw[3], 4: args.cw[4],
                                  5: args.cw[5], 0: args.cw[0]}

    # print params
    print("Parameters: \n", json.dumps(params, indent=4))
    # save params as json files
    with open(f'{params["out_dir"]}/train_{keyword}.json', 'w') as file:
        json.dump(params, file, indent=4)

    # initialize training object and train model
    train = training.TrainSegmentation(**params)
    train.train_model()
