"""
Training Fully Convolutional Networks
"""

import os
import glob
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, JaccardIndex, ConfusionMatrix
from tqdm import tqdm
from skimage.segmentation import watershed

# custom imports
from pipelines.datagen import datagen, tensor_datagen
from pipelines.models.load_models import load_model, load_optimizer
from pipelines.training.loss_functions import estimate_loss
from utils.common_utils import create_multi_folder, create_folder
from utils.tensorboard_plots import write_cm


def add_class_weights(weigth_dict, weight_batch, label_batch):
    # Add weights to the batch based on the class including background
    for label_loc, weight in weigth_dict.items():
        if label_loc == 0:
            weight_batch[(label_batch == label_loc) & (weight_batch > 0)] =\
                weight_batch[(label_batch == label_loc) & (weight_batch > 0)] + weight - 1
        else:
            weight_batch[(label_batch == label_loc) & (weight_batch == 1)] = weight
    return weight_batch


# function for estimating absolute difference between two batches
def abs_diff(y_true, y_pred):
    # count objects in the true and pred batch, get abs diff between them
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    true_count, pred_count = 0, 0
    for i in range(y_true.shape[0]):
        true_array = y_true[i].astype(np.uint16)
        pred_array = y_pred[i].astype(np.uint16)
        true_shed = watershed(-true_array, connectivity=2, mask=true_array > 0)
        pred_shed = watershed(-pred_array, connectivity=2, mask=pred_array > 0)
        true_count += len(np.unique(true_shed)) - 1
        pred_count += len(np.unique(pred_shed)) - 1
    return abs(true_count - pred_count)


class TrainSegmentation:
    def __init__(self, model_name, data_path, optimizer='adamw', learning_rate=None, batch_size_train=1,
                 batch_size_val=1, num_classes=2, patch_size=None, model_weights=None,
                 sample_size=None, epochs=10, image_folder="images", label_folder="p2_labels", weight_folder=None,
                 keyword="test", out_dir="../../runs/test", log_dir="../../runs/test/logs",
                 train_folder="train", val_folder="valid", border_threshold=None, label_list_=None, weights_=None,
                 use_border_weight=False, load_data2memory=False, load_data2device='cpu', tensordata_folder=None,
                 encoder_weights=None, weights_dict=None, loss_type='cross_entropy', t_params=(0.7, 0.3, 1.0)):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.model_weights = model_weights
        self.sample_size = sample_size
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.weight_folder = weight_folder
        self.epochs = epochs
        self.keyword = keyword
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.model_name = model_name
        self.optimizer = optimizer
        self.border_threshold = border_threshold
        self.label_list_ = label_list_
        self.weights_ = weights_
        self.use_border_weight = use_border_weight
        self.load_data2memory = load_data2memory
        self.load_data2device = load_data2device
        self.tensordata_folder = tensordata_folder
        self.encoder_weights = encoder_weights
        self.weights_dict = weights_dict
        self.loss_type = loss_type
        self.t_params = t_params

        # get device and print device info
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                print(f"GPU {i}: Name = {gpu_name}, Total Memory = {gpu_memory:.0f} MB")
        else:
            self.device = torch.device('cpu')

    def create_datasets(self, data_folder, set_type="train"):
        global weight_paths  # fixme remove this line later
        print(f"Data Folder: {data_folder}")
        image_paths = np.array(glob.glob(data_folder + f"/{self.image_folder}/*.tif"))
        label_paths = np.array(glob.glob(data_folder + f"/{self.label_folder}/*.tif"))
        image_paths.sort()
        label_paths.sort()
        if self.weight_folder is not None:
            weight_paths = np.array(glob.glob(data_folder + f"/{self.weight_folder}/*.tif"))
            weight_paths.sort()
            if self.sample_size is not None:
                weight_paths = weight_paths[:self.sample_size]
        else:
            weight_paths = None

        if self.sample_size is not None:
            image_paths = image_paths[:self.sample_size]
            label_paths = label_paths[:self.sample_size]
        print(f"No. of images: {len(image_paths)}, No. of labels {len(label_paths)}, No. of wieght"
              f" masks {len(weight_paths) if self.weight_folder is not None else 0}")
        assert len(image_paths) == len(label_paths) != 0, (f"Number of images({len(image_paths)}) "
                                                           f"!= labels ({len(label_paths)}) == 0")

        if self.load_data2memory:
            if self.tensordata_folder is not None and not os.path.exists(self.tensordata_folder):
                create_folder(self.tensordata_folder)
            file_folder = os.path.join(self.tensordata_folder, set_type)
            if not os.path.exists(file_folder):
                os.mkdir(file_folder)
            print(f"File Folder: {file_folder}")
            data = tensor_datagen.TensorDataGenerator(image_paths, label_paths, self.num_classes, weight_paths,
                                                      batch_size=self.batch_size_train,
                                                      patch_size=self.patch_size,
                                                      device=self.device, border_threshold=self.border_threshold,
                                                      label_list_=self.label_list_, weights_=self.weights_,
                                                      use_border_weight=self.use_border_weight,
                                                      data_device=self.load_data2device,
                                                      output_folder=file_folder)
        else:
            data = datagen.DataGenerator(image_paths, label_paths, self.num_classes, weight_paths,
                                         batch_size=self.batch_size_train, patch_size=self.patch_size,
                                         device=self.device, border_threshold=self.border_threshold,
                                         label_list_=self.label_list_, weights_=self.weights_,
                                         use_border_weight=self.use_border_weight)
        return data

    def train_model(self):
        # data paths
        train_path = os.path.join(*[self.data_path, self.train_folder])
        valid_path = os.path.join(*[self.data_path, self.val_folder])

        # data loaders
        train_data = self.create_datasets(train_path, set_type="train")
        val_data = self.create_datasets(valid_path, set_type="valid")
        train_loader = DataLoader(train_data, self.batch_size_train, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, self.batch_size_val)

        # model
        model = load_model(self.model_name, self.device, channels=3, num_classes=self.num_classes,
                           pre_trained_weight=self.model_weights, patch_size=self.patch_size,
                           encoder_weights=self.encoder_weights)

        # intialize logs and metrics
        outmodel_dir = os.path.join(*[self.out_dir, self.keyword])
        fidx = 1
        while os.path.exists(outmodel_dir):
            outmodel_dir = os.path.join(*[self.out_dir, f"{self.keyword}_{fidx}"])
            fidx += 1
        create_multi_folder(outmodel_dir)
        if self.keyword != "test":  # to avoid mutliple tensorboard folders for test
            tensorboard_folder = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.keyword}"
        else:
            tensorboard_folder = self.keyword
        print(f"Tensorboard Folder: {tensorboard_folder}")
        writer = torch.utils.tensorboard.SummaryWriter(f"{self.log_dir}/{tensorboard_folder}")

        # training
        if 'ordinal' in self.loss_type:
            task = "binary"
            metric_class = 1
        elif self.num_classes > 1:
            task = "multiclass"
            metric_class = self.num_classes
        else:
            task = "binary"
            metric_class = 1

        iou_m = JaccardIndex(task=task, num_classes=metric_class, absent_score=0).to(self.device)
        conf = ConfusionMatrix(num_classes=metric_class, task=task).to(self.device)
        accuracy_m = Accuracy(task=task, top_k=1, num_classes=metric_class).to(self.device)
        best_iou = -1
        best_mae = 10e10

        optimizer = load_optimizer(self.optimizer, model, learning_rate=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        for epoch in range(self.epochs):
            model.train()
            loss_avg = 0
            accuracy_m.reset()
            iou_m.reset()
            conf.reset()

            pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc="train")
            for i, batch in pbar_train:  # fixme use with torch.cuda.amp.autocast():
                X = batch['X']
                y = batch['y']
                if self.weight_folder is not None:
                    w = batch['w'].float()
                    if self.weights_dict is not None:
                        # fixme create new weight mask using weight_dict if weight_folder is None
                        w = add_class_weights(self.weights_dict, w, y)
                else:
                    w = None
                if self.model_name == 'dinov2':  # dino v2 model takes only 448 size image (need to explote)
                    X = F.interpolate(X, size=(448, 448), mode='bilinear')
                    y = F.interpolate(y.float(), size=(448, 448), mode='nearest').long()
                    w = F.interpolate(w, size=(448, 448), mode='nearest')

                X, y_, w = train_data.augment(X, y, self.device, w)
                pred = model(X)
                loss = estimate_loss(pred, y_, self.model_name, w.squeeze(1), loss_type=self.loss_type,
                                     num_classes=self.num_classes, t_params=self.t_params)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()

                # metric update
                data_mask = w > 0
                if self.num_classes == 1:  # binary segmentation
                    pred_labels = pred >= 0.0                           # y_ is augmented label
                    _y, _pred_labels = y_, pred_labels                  # _y and _pred_labels is for metrics
                else:                                                   # pred_labels is for tensorboard visualization
                    if 'ordinal' in self.loss_type:  # ordinal segmentation
                        pred_labels = (pred > 0.0).float().cumprod(1).sum(1, keepdim=True) * data_mask
                        mask = y_ != 0
                        _y, _pred_labels = y_[mask]-1, pred_labels[mask]-1
                        _pred_labels[_pred_labels == -1] = 0
                        _pred_labels, _y = (_pred_labels > 0).long(), (_y > 0).long()

                    else:   # multiclass segmentation
                        pred_labels = pred.argmax(dim=1, keepdim=True) * data_mask
                        _y, _pred_labels = y_.squeeze(1), pred_labels.squeeze(1)

                if _pred_labels.sum() > 0:
                    accuracy_m.update(_pred_labels, _y)
                    conf.update(_pred_labels.flatten().to(self.device), _y.flatten().to(self.device))
                    iou_m.update(_pred_labels, _y)

                if torch.cuda.is_available():  # Check if CUDA is available
                    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    pbar_train.set_postfix(loss=f'{loss_avg / (i + 1):.4f}',
                                           current_mem=f'{current_memory:.2f} MB',
                                           peak_mem=f'{peak_memory:.2f} MB',
                                           iou=f'{iou_m.compute():.4f}',
                                           epoch=f'{epoch + 1}/{self.epochs}')
                else:
                    pbar_train.set_postfix(loss=f'{loss_avg / (i + 1):.4f}', iou=f'{iou_m.compute():.4f}',
                                           epoch=f'{epoch + 1}/{self.epochs}')
            pbar_train.reset()

            # log image and weights
            b_, c_, h_, w_ = X.shape
            if c_ > 3:  # first three channels considered as rgb
                X = X[:, :3]  # if image channels are more than three
            writer.add_images("train/input_rgb", X, global_step=epoch, dataformats="NCHW")
            if self.weight_folder is not None:
                writer.add_images("train/weights", (w - w.min()) / (w.max() - w.min()), global_step=epoch,
                                  dataformats="NCHW")
            writer.add_images(f"train/label", y_ / self.num_classes, global_step=epoch, dataformats="NCHW")
            writer.add_images(f"train/pred", pred_labels / self.num_classes, global_step=epoch,
                              dataformats="NCHW")
            # log scalar/metrics values
            writer.add_scalar("train/loss", loss_avg / len(train_loader), global_step=epoch)
            writer.add_scalar(f"train/acc", accuracy_m.compute(), global_step=epoch)
            writer.add_scalar(f"train/iou", iou_m.compute(), global_step=epoch)
            write_cm(writer, metric_class, conf.compute())

            # validation
            model.eval()
            accuracy_m.reset()
            iou_m.reset()
            conf.reset()
            loss_avg = 0
            mae = 0
            pbar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc="val")
            for i, batch in pbar_val:
                X = batch['X']
                y = batch['y']
                if self.weight_folder is not None:
                    w = batch['w'].float()
                    if self.weights_dict is not None:
                        w = add_class_weights(self.weights_dict, w, y)
                else:
                    w = None
                with torch.no_grad():
                    if self.model_name == 'dinov2':  # dino v2 model takes only 448 size image (need to explote)
                        X = F.interpolate(X, size=(448, 448), mode='bilinear')
                        y = F.interpolate(y.float(), size=(448, 448), mode='nearest').long()
                        w = F.interpolate(w, size=(448, 448), mode='nearest')
                    pred = model(X)
                    loss = estimate_loss(pred, y, self.model_name, w.squeeze(1), loss_type=self.loss_type,
                                         num_classes=self.num_classes, t_params=self.t_params)
                    loss_avg += loss.item()

                    data_mask = w > 0
                    # metric update
                    if self.num_classes == 1:
                        pred_labels = pred >= 0.0
                        _y, _pred_labels = y, pred_labels
                    else:
                        if 'ordinal' in self.loss_type:
                            pred_labels = (pred >= 0.0).cumprod(1).sum(1, keepdim=True) * data_mask
                            mask = y != 0
                            _y, _pred_labels = y[mask] - 1, pred_labels[mask] - 1
                            _pred_labels[_pred_labels == -1] = 0
                            _pred_labels, _y = (_pred_labels > 0).long(), (_y > 0).long()
                            mae += abs_diff(y, pred_labels)
                        else:
                            pred_labels = pred.argmax(dim=1, keepdim=True) * data_mask
                            _y, _pred_labels = y.squeeze(1), pred_labels.squeeze(1)

                    if _pred_labels.sum() > 0:
                        accuracy_m.update(_pred_labels, _y)
                        conf.update(_pred_labels.flatten().to(self.device), _y.flatten().to(self.device))
                        iou_m.update(_pred_labels, _y)

                    pbar_val.set_postfix(loss=f'{loss_avg / (i + 1):.4f}', iou=f'{iou_m.compute():.4f}')
            pbar_val.reset()

            # log image and weights
            if c_ > 3:  # first three channels considered as rgb
                X = X[:, :3]  # if image channels are more than three
            writer.add_images("val/input_rgb", X, global_step=epoch, dataformats="NCHW")
            if self.weight_folder is not None:
                writer.add_images("val/weights", (w - w.min()) / (w.max() - w.min()), global_step=epoch,
                                  dataformats="NCHW")
            # log multi-label input/outputs
            writer.add_images(f"val/label", y / self.num_classes, global_step=epoch, dataformats="NCHW")
            writer.add_images(f"val/pred", pred_labels / self.num_classes,
                              global_step=epoch, dataformats="NCHW")

            # log scalar/metrics values
            iou = iou_m.compute()
            writer.add_scalar("val/loss", loss_avg / len(val_loader), global_step=epoch)
            writer.add_scalar(f"val/acc", accuracy_m.compute(), global_step=epoch)
            writer.add_scalar(f"val/iou", iou, global_step=epoch)
            if 'ordinal' in self.loss_type:
                writer.add_scalar("val/mae", mae, global_step=epoch)
            write_cm(writer, metric_class, conf.compute(), set_type='val')

            # save model
            if 'ordinal' in self.loss_type:
                if best_mae > mae and best_iou < iou:   # fixme clean duplicate lines
                    best_mae, best_iou = mae, iou
                    torch.save(
                        model.state_dict(),
                        outmodel_dir + f"/best_model"
                    )
                    # save metrics at every best epoch to single txt file
                    with open(outmodel_dir + f"/best_epochs_metrics_{self.keyword}.txt", "a") as file:
                        file.write(f"Epoch: {epoch + 1}, Best iou: {best_iou}, Best mae: {best_mae}\n")
            else:
                if best_iou < iou:
                    best_iou = iou
                    torch.save(
                        model.state_dict(),
                        outmodel_dir + f"/best_model"
                    )
                    # save metrics at every best epoch to single txt file
                    with open(outmodel_dir + f"/best_epochs_metrics_{self.keyword}.txt", "a") as file:
                        file.write(f"Epoch: {epoch + 1}, Best iou: {best_iou}\n")

            # save last epoch model with epoch number in it
            torch.save(model.state_dict(), outmodel_dir + f"/last_epoch_{epoch + 1}")
            # delete previous model
            if epoch > 0:
                os.remove(outmodel_dir + f"/last_epoch_{epoch}")

            # scheduler step
            scheduler.step()
