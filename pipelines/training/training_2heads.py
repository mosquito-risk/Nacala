"""
Training Fully Convolutional Networks
"""

import os
import glob
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, JaccardIndex, ConfusionMatrix
from tqdm import tqdm
from scipy.ndimage import label

# custom imports
from pipelines.datagen import datagen_2heads as datagen
from pipelines.datagen import tensor_datagen_2heads
from pipelines.models.load_models import load_model, load_optimizer
from pipelines.training.loss_functions import estimate_loss
from utils.common_utils import create_multi_folder, create_folder
from utils.tensorboard_plots import write_cm


def abs_diff_2heads(y_true, y_pred):
    # count objects in the true and pred batch, get abs diff between them
    y_true = y_true.detach().cpu().numpy()[:, [1], :, :]
    y_pred = y_pred.detach().cpu().numpy()[:, [1], :, :]
    true_count, pred_count = 0, 0
    for i in range(y_true.shape[0]):
        _, num_true = label(y_true)
        true_count += num_true
        _, num_pred = label(y_pred)
        pred_count += num_pred
    return abs(true_count - pred_count)


class TrainSegmentation:
    def __init__(self, model_name, data_path, optimizer='adamw', learning_rate=None, batch_size_train=1,
                 batch_size_val=1, num_classes=1, patch_size=None, model_weights=None, sample_size=None,
                 epochs=10, image_folder="images", label1_folder="p2_labels", label2_folder=None,
                 weight_folder=None, keyword="test", load_data2memory=False, tensordata_folder=None,
                 out_dir="../../runs/test", log_dir="../../runs/test/logs", train_folder="train", val_folder="valid",
                 load_data2device='cpu', encoder_weights=None, loss_type="cross_entropy", head_size='n',
                 t_params=(0.7, 0.3, 1.0)):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.data_path = data_path
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.model_weights = model_weights
        self.sample_size = sample_size
        self.image_folder = image_folder
        self.label1_folder = label1_folder
        self.label2_folder = label2_folder
        self.weight_folder = weight_folder  # fixme mask pixels has to be in weight masks
        self.epochs = epochs
        self.keyword = keyword
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.model_name = model_name
        self.optimizer = optimizer
        self.load_data2memory = load_data2memory
        self.tensordata_folder = tensordata_folder
        self.load_data2device = load_data2device
        self.loss_type = loss_type
        self.encoder_weights = encoder_weights
        self.head_size = head_size
        self.t_params = t_params
        self.num_outputs = 2

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
        image_paths = np.array(glob.glob(data_folder + f"/{self.image_folder}/*.tif"))
        label1_paths = np.array(glob.glob(data_folder + f"/{self.label1_folder}/*.tif"))
        label2_paths = np.array(glob.glob(data_folder + f"/{self.label2_folder}/*.tif"))
        image_paths.sort()
        label1_paths.sort()
        label2_paths.sort()
        if self.weight_folder is not None:
            weight_paths = np.array(glob.glob(data_folder + f"/{self.weight_folder}/*.tif"))
            weight_paths.sort()
            if self.sample_size is not None:
                weight_paths = weight_paths[:self.sample_size]
        else:
            weight_paths = None

        if self.sample_size is not None:
            image_paths = image_paths[:self.sample_size]
            label1_paths = label1_paths[:self.sample_size]
            label2_paths = label2_paths[:self.sample_size]
        print(f"No. of images: {len(image_paths)}, No. of labels-1 {len(label1_paths)},"
              f" No. of labels-2 {len(label2_paths)}, No. of wieght"
              f" masks {len(weight_paths) if self.weight_folder is not None else 0}")
        assert len(image_paths) == len(label2_paths) == len(label2_paths) != 0, "Number of patches/images are not equal"
        if self.load_data2memory:
            if self.tensordata_folder is not None and not os.path.exists(self.tensordata_folder):
                create_folder(self.tensordata_folder)
            file_folder = os.path.join(self.tensordata_folder, set_type)
            data = tensor_datagen_2heads.TensorDataGenerator(image_paths, label1_paths, label2_paths,
                                                             weight_paths, batch_size=self.batch_size_train,
                                                             patch_size=self.patch_size,
                                                             device=self.device,
                                                             data_device=self.load_data2device,
                                                             output_folder=file_folder)
        else:
            data = datagen.DataGenerator(image_paths, label1_paths, label2_paths, weight_paths,
                                         batch_size=self.batch_size_train, patch_size=self.patch_size,
                                         device=self.device)
        return data

    def train_model(self):
        # data
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
                           encoder_weights=self.encoder_weights, head_size=self.head_size)
        print(f"Model: {model}")

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

        # metrics
        iou_ms = [JaccardIndex(task="binary", num_classes=self.num_classes, absent_score=0).to(self.device)
                  for _ in range(self.num_outputs)]
        confs = [ConfusionMatrix(num_classes=self.num_classes, task="binary").to(self.device)
                 for _ in range(self.num_outputs)]
        accuracy_ms = [Accuracy(task="binary", top_k=1, num_classes=self.num_classes
                                ).to(self.device)
                       for _ in range(self.num_outputs)]

        # training
        best_iou = -1
        best_mae = 10e10
        optimizer = load_optimizer(self.optimizer, model, learning_rate=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        for epoch in range(self.epochs):
            model.train()
            loss_avg = 0
            pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc="train")
            for i, batch in pbar_train:  # fixme use with torch.cuda.amp.autocast():
                X = batch['X']
                y = batch['y']
                if self.weight_folder is not None:
                    w = batch['w']
                else:
                    w = None
                if self.model_name == 'dinov2':
                    X = F.interpolate(X, size=(448, 448), mode='bilinear')
                    y = F.interpolate(y.float(), size=(448, 448), mode='nearest').long()
                    w = F.interpolate(w.float(), size=(448, 448), mode='nearest').long()

                X, y_, w = train_data.augment(X, y, self.device, w)
                pred_ = model(X)
                loss = estimate_loss(pred_, y_, self.model_name, w.squeeze(1), loss_type=self.loss_type,
                                     num_classes=self.num_classes, t_params=self.t_params)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()

                # metric update
                data_mask = (w > 0).float()
                for idx in range(self.num_outputs):  # for multi-head/decoder models
                    pred = pred_[:, idx, :, :] * data_mask.squeeze(1)
                    single_y = y_[:, idx, :, :]
                    pred_labels = (pred >= 0).float()
                    accuracy_ms[idx].update(pred, single_y)
                    iou_ms[idx].update(pred_labels, single_y)
                    confs[idx].update(pred_labels.flatten().to(self.device), single_y.flatten().to(self.device))

                iou_mean = sum([iou_ms[idx].compute() for idx in range(self.num_outputs)]) / self.num_outputs
                if torch.cuda.is_available():  # Check if CUDA is available
                    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    pbar_train.set_postfix(loss=f'{loss_avg / (i + 1):.4f}',
                                           current_mem=f'{current_memory:.2f} MB',
                                           peak_mem=f'{peak_memory:.2f} MB',
                                           iou0=f'{iou_ms[0].compute():.4f}',
                                           iou1=f'{iou_ms[1].compute():.4f}',
                                           iou_mean=f'{iou_mean:.4f}',
                                           epoch=f'{epoch + 1}/{self.epochs}')
                else:
                    pbar_train.set_postfix(loss=f'{loss_avg / (i + 1):.4f}', iou0=f'{iou_ms[0].compute():.4f}',
                                           iou1=f'{iou_ms[1].compute():.4f}', epoch=f'{epoch + 1}/{self.epochs}')
            pbar_train.reset()

            # log image and weights
            b_, c_, h_, w_ = X.shape
            if c_ > 3:  # first three channels considered as rgb
                X = X[:, :3]  # if image channels are more than three
            writer.add_images("train/input_rgb", X, global_step=epoch, dataformats="NCHW")
            if self.weight_folder is not None:
                writer.add_images("train/weights", (w - w.min()) / (w.max() - w.min()), global_step=epoch,
                                  dataformats="NCHW")
            # log multi-label input/outputs
            for j in range(self.num_outputs):
                writer.add_images(f"train/label_{j}", y_[:, [j], :, :] / self.num_classes, global_step=epoch,
                                  dataformats="NCHW")
            for idx in range(self.num_outputs):
                pred_v = (pred_[:, [idx], :, :] >= 0).float() * data_mask
                writer.add_images(f"train/pred_{idx}", pred_v / self.num_classes,
                                  global_step=epoch, dataformats="NCHW")

            # calculate mean iou
            iou = sum([iou_ms[idx].compute() for idx in range(self.num_outputs)]) / self.num_outputs
            writer.add_scalar(f"train/iou_mean", iou, global_step=epoch)

            # log scalar/metrics values
            writer.add_scalar("train/loss", loss_avg / len(train_loader), global_step=epoch)
            for idx in range(self.num_outputs):
                writer.add_scalar(f"train/acc_{idx}", accuracy_ms[idx].compute(), global_step=epoch)
                accuracy_ms[idx].reset()
                writer.add_scalar(f"train/iou_{idx}", iou_ms[idx].compute(), global_step=epoch)
                iou_ms[idx].reset()
                write_cm(writer, self.num_classes, confs[idx].compute(), 'train', idx)
                confs[idx].reset()

            # validation
            model.eval()
            loss_avg = 0
            mae = 0
            pbar_val = tqdm(enumerate(val_loader), desc="val")
            for i, batch in pbar_val:
                X = batch['X']
                y_ = batch['y']
                if self.weight_folder is not None:
                    w = batch['w']
                else:
                    w = None
                with torch.no_grad():
                    if self.model_name == 'dinov2':
                        X = F.interpolate(X, size=(448, 448), mode='bilinear')
                        y_ = F.interpolate(y_.float(), size=(448, 448), mode='nearest').long()
                        w = F.interpolate(w.float(), size=(448, 448), mode='nearest').long()
                    pred_ = model(X)
                    loss = estimate_loss(pred_, y_, self.model_name, w.squeeze(1), loss_type=self.loss_type,
                                         num_classes=self.num_classes,
                                         t_params=self.t_params)
                    loss_avg += loss.item()

                    # metric update
                    data_mask = (w > 0).float()
                    for idx in range(self.num_outputs):
                        pred = pred_[:, idx, :, :] * data_mask.squeeze(1)
                        single_y = y_[:, idx, :, :]
                        pred_labels = (pred >= 0).float()
                        accuracy_ms[idx].update(pred, single_y)
                        iou_ms[idx].update(pred_labels, single_y)
                        confs[idx].update(pred_labels.flatten().to(self.device), single_y.flatten().to(self.device))

                    mae += abs_diff_2heads(y_, (pred_ >= 0).float())
                    pbar_val.set_postfix(loss=f'{loss_avg / (i + 1):.4f}', iou0=f'{iou_ms[0].compute():.4f}',
                                         iou1=f'{iou_ms[1].compute():.4f}')
            pbar_val.reset()

            # log image and weights
            if c_ > 3:  # first three channels considered as rgb
                X = X[:, :3]  # if image channels are more than three
            writer.add_images("val/input_rgb", X, global_step=epoch, dataformats="NCHW")
            if self.weight_folder is not None:
                writer.add_images("val/weights", (w - w.min()) / (w.max() - w.min()), global_step=epoch,
                                  dataformats="NCHW")
            # log multi-label input/outputs
            for j in range(self.num_outputs):
                writer.add_images(f"val/label_{j}", y_[:, [j], :, :] / self.num_classes, global_step=epoch,
                                  dataformats="NCHW")
            for idx in range(self.num_outputs):
                pred_v = (pred_[:, [idx], :, :] >= 0).float() * data_mask
                writer.add_images(f"val/pred_{idx}", pred_v / self.num_classes,
                                  global_step=epoch, dataformats="NCHW")

            # calculate mean iou
            iou = sum([iou_ms[idx].compute() for idx in range(self.num_outputs)]) / self.num_outputs
            if self.num_outputs > 1:
                writer.add_scalar(f"val/iou_mean", iou, global_step=epoch)

            # log scalar/metrics values
            writer.add_scalar("val/loss", loss_avg / len(val_loader), global_step=epoch)
            writer.add_scalar("val/mae", mae, global_step=epoch)
            for idx in range(self.num_outputs):
                writer.add_scalar(f"val/acc_{idx}", accuracy_ms[idx].compute(), global_step=epoch)
                writer.add_scalar(f"val/iou_{idx}", iou_ms[idx].compute(), global_step=epoch)
                write_cm(writer, self.num_classes, confs[idx].compute(), 'val', idx)
                accuracy_ms[idx].reset()
                iou_ms[idx].reset()
                confs[idx].reset()

            # save model
            if best_iou < iou:
                best_mae, best_iou = mae, iou
                torch.save(
                    model.state_dict(),
                    outmodel_dir + f"/best_model"
                )
                # save metrics at every best epoch to single txt file
                with open(outmodel_dir + f"/best_epochs_metrics_{self.keyword}.txt", "a") as file:
                    file.write(f"Epoch: {epoch + 1}, Best iou: {best_iou}, Best mae: {best_mae}\n")

            # save last epoch model with epoch number in it
            torch.save(model.state_dict(), outmodel_dir + f"/last_epoch_{epoch + 1}")
            # delete previous model
            if epoch > 0:
                os.remove(outmodel_dir + f"/last_epoch_{epoch}")

            # scheduler step
            scheduler.step()
