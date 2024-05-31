import torch
import torch.nn.functional as F
from pipelines import base_class
from segmentation_models_pytorch import Unet
from pipelines.models import dinov2_seg


def decision_tensor(tensor1, tensor2, operator):
    if operator == 'MIN':
        resultant = torch.minimum(tensor1, tensor2)
    elif operator == 'MAX':
        resultant = torch.maximum(tensor1, tensor2)
    elif operator == 'MEAN':
        # fixme: mean is calculating over zeros on one of the tensors
        resultant = (tensor1 + tensor2) / 2
    else:  # operator == 'REPLACE':
        resultant = tensor2
    return resultant


class ModelsClass(base_class.BaseClass):
    def __init__(self, model_name, num_classes=2, channels=3, weights_path=None, device='cpu', image_dir=None,
                 label_dir=None):
        super().__init__(image_dir, label_dir)
        self.model_name = model_name
        self.channels = channels
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.device = device

    def load_model(self):  # fixme: this method is not used
        if self.model_name == 'unet':
            model = Unet(in_channels=self.channels, classes=self.num_classes)
        elif self.model_name == 'uresnet18':
            model = Unet(in_channels=self.channels, classes=self.num_classes, encoder_name='resnet18')
        elif self.model_name == "delfors_unet":
            model = DeLfoRSUNet(n_classes=self.num_classes, in_channels=self.channels, backbone='standard',
                                activation='elu',
                                patch_size=(512, 512))
        elif self.model_name == "dinov2":
            id2label = list(range(self.num_classes))
            model = dinov2_seg.Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",
                                                                             id2label=id2label,
                                                                             num_labels=self.num_classes)
            for name, param in model.named_parameters():
                if name.startswith("dinov2"):
                    param.requires_grad = False
        else:
            raise NotImplementedError(f' {self.model_name} is not implemeted...')
        if self.weights_path is not None:
            model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        return model.to(self.device).eval()

    def predict_and_replace(self, batch, batch_pos, mask, operator, num_cls=1, loss_type=None):
        """
        Predict one batch of patches with tensorflow, and add result to the output prediction.
        """
        tm = batch / 255.
        if self.model_name == 'delfors_unet':
            predictions = self.model(tm)
        elif self.model_name == 'dinov2':
            tm = F.interpolate(tm, size=(448, 448), mode='bilinear')
            predictions = self.model(tm)
            predictions = F.interpolate(predictions, size=(512, 512), mode='bilinear')
        else:
            predictions = self.model.predict(tm)
        if num_cls == 1:
            predictions = torch.sigmoid(predictions)
        else:
            if loss_type == 'ordinal':
                predictions = torch.sigmoid(predictions)
            else:
                predictions = F.softmax(predictions, dim=1)

        # remove prediction in nodata areas
        channel_sum = torch.sum(batch, dim=1)  # Sum across channels
        zero_mask = (channel_sum != 0).expand_as(predictions)
        predictions = (predictions*zero_mask).detach().cpu()

        for i, (col, row, wi, he) in enumerate(batch_pos):
            p = predictions[i, :, :he, :wi]
            sub_mask = mask[:, row:row + he, col:col + wi]
            resultant = decision_tensor(sub_mask, p, operator)
            mask[:, row:row + he, col:col + wi] = resultant
        return mask

    def predict_and_replace_multi(self, batch, batch_pos, masks: list, operator):
        """
        Predict one batch of patches and add result to the output prediction.
        """
        tm = batch / 255
        predictions = self.model.predict(tm)
        # remove prediction in nodata areas
        channel_sum = torch.sum(batch, dim=1)  # Sum across channels

        bat_, out_, _, _ = predictions.shape
        for idx, out_idx in enumerate(range(out_)):
            pred = predictions[:, [out_idx], :, :]
            pred = torch.sigmoid(pred)
            zero_mask = (channel_sum != 0).expand_as(pred)
            pred = (pred * zero_mask).detach().cpu()
            # pred = F.softmax(pred, dim=1).detach().cpu()  # completely switching to single logit output mode
            mask = masks[idx]
            for i, (col, row, wi, he) in enumerate(batch_pos):
                p = pred[i, :, :he, :wi]
                sub_mask = mask[:, row:row + he, col:col + wi]
                resultant = decision_tensor(sub_mask, p, operator)
                mask[:, row:row + he, col:col + wi] = resultant
            masks[idx] = mask
        return masks

    def predict_and_replace_multi_cls(self, batch, batch_pos, masks: list, operator):
        """
        Predict one batch of patches and add result to the output prediction.
        """
        tm = batch / 255
        predictions = self.model.predict(tm)
        # remove prediction in nodata areas
        channel_sum = torch.sum(batch, dim=1)  # Sum across channels

        # Binary mask
        pred1 = (predictions[:, [0], :, :] >= 0).int()
        zero_mask1 = (channel_sum != 0).expand_as(pred1)
        pred1 = (pred1 * zero_mask1).detach().cpu()
        mask = masks[0]
        for i, (col, row, wi, he) in enumerate(batch_pos):
            p = pred1[i, :, :he, :wi]
            sub_mask = mask[:, row:row + he, col:col + wi]
            resultant = decision_tensor(sub_mask, p, operator)
            mask[:, row:row + he, col:col + wi] = resultant
        masks[0] = mask

        # Multiclass mask
        pred2 = F.softmax(predictions[:, 1:, :, :], dim=1)
        zero_mask2 = (channel_sum != 0).expand_as(pred2)
        pred2 = (pred2 * zero_mask2).detach().cpu()
        mask = masks[1]
        for i, (col, row, wi, he) in enumerate(batch_pos):
            p = pred2[i, :, :he, :wi]
            sub_mask = mask[:, row:row + he, col:col + wi]
            resultant = decision_tensor(sub_mask, p, operator)
            mask[:, row:row + he, col:col + wi] = resultant
        masks[1] = mask
        return masks
