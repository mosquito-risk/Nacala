"""
Loss functions for traning fully convolutional networks
"""
import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot
from segmentation_models_pytorch.losses import TverskyLoss
from pipelines.training.tversky_loss import BinaryTverskyLoss, BinaryTverskyLoss_v2


def ordinal_labels(y_true, num_classes):
    """
    Convert ordinal labels from energy map labels
    :param y_true: Energy map labels with some distance
    :param num_classes: Total number of energy levels
    :return: Ordinal labels
    """
    y_energy = one_hot(y_true, num_classes=num_classes + 1)[..., 1:]
    y_energy = y_energy.flip(-1)
    y_energy.cumsum_(-1)
    y_energy = y_energy.flip(-1)
    y_energy = y_energy.moveaxis(-1, 1)
    return y_energy.float()


def estimate_loss(model_pred, ground_truth, model_name, weights=None, loss_type='cross_entropy',
                  num_classes=2, t_params: tuple = (0.7, 0.3, 1.0), smooth_factor=0.1):
    alpha, beta, gamma = t_params
    weights = weights.squeeze(1) if weights is not None else None
    if model_name == 'unet_2heads' or model_name == 'unet_2decoders' \
            or model_name == 'dinov2_2heads':
        # Cross entropy loss
        if loss_type == 'cross_entropy':
            criterion = F.binary_cross_entropy_with_logits
            ground_truth = (1 - ground_truth) * smooth_factor + ground_truth * (1 - smooth_factor)
            loss = criterion(model_pred, ground_truth.float(), reduction="none")
            if weights is not None:
                loss1 = (loss[:, 0, :, :] * (weights > 0).float()).sum().div((weights > 0).sum())
                loss2 = (loss[:, 1, :, :] * weights).sum().div((weights > 0).sum())
                loss = (loss1 + loss2) / 2
            else:
                loss = loss.mean()

        elif loss_type == 'cross_entropy_cls':
            criterion1 = F.binary_cross_entropy_with_logits
            criterion2 = F.cross_entropy
            ground_truth1 = ((1 - ground_truth[:, [0], :, :]) * smooth_factor + ground_truth[:, [0], :, :]
                             * (1 - smooth_factor))
            loss1 = criterion1(model_pred[:, [0], :, :], ground_truth1.float(), reduction="none").squeeze(1)
            loss2 = criterion2(model_pred[:, 1:, :, :], ground_truth[:, 1, :, :], reduction="none", label_smoothing=0.1)
            if weights is not None:
                loss1 = (loss1 * (weights > 0).float()).sum().div((weights > 0).sum())
                loss2 = (loss2 * weights).sum().div((weights > 0).sum())
                loss = (loss1 + loss2) / 2
            else:
                loss = (loss1.mean() + loss2.mean()) / 2

        elif loss_type == 'cross_entropy_cls3':
            criterion = F.cross_entropy
            loss1 = criterion(model_pred[:, :6, :], ground_truth[:, 0, :, :], reduction="none", label_smoothing=0.1)
            loss2 = criterion(model_pred[:, 6:, :, :], ground_truth[:, 1, :, :], reduction="none", label_smoothing=0.1)
            if weights is not None:
                loss1 = (loss1 * weights).sum().div((weights > 0).sum())
                loss2 = (loss2 * weights).sum().div((weights > 0).sum())
                loss = (loss1 + loss2) / 2
            else:
                loss = (loss1.mean() + loss2.mean()) / 2

        # Tversky loss
        elif loss_type == 'tversky':
            # criterion = TverskyLoss(mode='binary', alpha=alpha, beta=beta, smooth=1e-6, gamma=gamma)
            criterion = BinaryTverskyLoss_v2(alpha=alpha, beta=beta, gamma=gamma)
            pred = torch.sigmoid(model_pred)
            if weights is not None:
                pred1 = pred[:, 0, :, :] * (weights > 0).float()
                pred2 = pred[:, 1, :, :] * weights
                pred = torch.stack([pred1, pred2], dim=1)
            loss = criterion(pred, ground_truth)

        else:
            raise NotImplementedError(f'Loss type {loss_type} is not implemented for {model_name}...')

    elif num_classes >= 2:
        # Cross entropy loss for Multi-class
        if loss_type == 'cross_entropy':
            if weights is not None:
                loss = (F.cross_entropy(model_pred, ground_truth.squeeze(1), reduction="none",
                                        label_smoothing=0.1) * weights).sum().div((weights > 0).sum())
            else:
                loss = F.cross_entropy(model_pred, ground_truth.squeeze(1))

        # Tversky loss for Multi-class
        elif loss_type == 'tversky':
            criterion = TverskyLoss(mode='multiclass', alpha=alpha, beta=beta, smooth=1e-6, gamma=gamma,
                                    from_logits=False)
            pred = torch.softmax(model_pred, dim=1)
            if weights is not None:
                pred = pred * weights.unsqueeze(1).expand_as(pred)
            loss = criterion(pred, ground_truth.squeeze(1))

        # Cross entropy for ordinal labels
        elif loss_type == 'ordinal_cross_entropy':
            ground_truth = ordinal_labels(ground_truth.squeeze(1), num_classes)
            loss = F.binary_cross_entropy_with_logits(model_pred, ground_truth, reduction='none')
            per_pixel_loss = loss.sum(dim=1)
            if weights is not None:
                loss = (per_pixel_loss * (weights > 0).float()).sum() / (weights > 0).sum()
            else:
                loss = per_pixel_loss.mean()

        # Tversky for ordinal labels
        elif loss_type == 'ordinal_tversky':
            # criterion = TverskyLoss(mode='binary', alpha=alpha, beta=beta, smooth=1e-6, gamma=gamma)
            criterion = BinaryTverskyLoss_v2(alpha=alpha, beta=beta, gamma=gamma)  #  there is v2
            ground_truth = ordinal_labels(ground_truth.squeeze(1), num_classes)
            pred = torch.sigmoid(model_pred)
            if weights is not None:
                pred = pred * weights.unsqueeze(1).expand_as(pred)
            loss = criterion(pred, ground_truth)

        # Ordinal cross entropy with weights (took it from random location :)
        elif loss_type == 'ordinal_weights':
            loss = ordinal_cross_entropy_with_weights(ground_truth.squeeze(1), model_pred, weights)

        else:
            raise NotImplementedError(f'Loss type {loss_type} is not implemented for {model_name}...')
    elif model_name == 'only_detector':
        if loss_type == 'cross_entropy':
            criterion = F.binary_cross_entropy_with_logits
            loss = criterion(model_pred, ground_truth.float(), reduction="none")

            # weight mask to multiply binary mask with 300
            weights = ground_truth * 300
            if weights is not None:
                loss = (loss * weights).sum().div((weights > 0).sum())
            else:
                loss = loss.mean()

    elif num_classes == 1:
        if loss_type == 'cross_entropy':
            criterion = F.binary_cross_entropy_with_logits
            ground_truth = (1 - ground_truth) * smooth_factor + ground_truth * (1 - smooth_factor)
            loss = criterion(model_pred, ground_truth.float(), reduction="none")
            if weights is not None:
                loss = (loss * weights).sum().div((weights > 0).sum())
            else:
                loss = loss.mean()

        # Tversky loss
        elif loss_type == 'tversky':
            # criterion = TverskyLoss(mode='binary', alpha=alpha, beta=beta, smooth=1e-6, gamma=gamma)
            criterion = BinaryTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
            pred = torch.sigmoid(model_pred)
            if weights is not None:
                pred = pred * (weights > 0).float()
            loss = criterion(pred, ground_truth)

        else:
            raise NotImplementedError(f'Loss type {loss_type} is not implemented for {model_name}...')

    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented for {model_name}...')

    return loss
