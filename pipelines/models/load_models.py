"""
List of models and optimizer to load
"""

import torch
from pipelines.models import dinov2_seg, unet_2heads, unet_2decoders, dinov2_seg_2heads
from segmentation_models_pytorch import Unet


def load_model(model_name, device, channels=3, num_classes=2, pre_trained_weight=None, patch_size=512,
               encoder_weights="imagenet", head_size='n', num_classes2=None):
    """
    Load model based on the model name
    :param model_name: Name of the model
    :param device: Device to load the model
    :param channels: Number of input channels
    :param num_classes: Number of classes
    :param pre_trained_weight: Pre-trained weight path
    :param patch_size: Patch size of the input
    :param encoder_weights: Encoder weights: Should be from ['imagenet', None]
    :param head_size: Head size for two heads model (only for unet_2heads or unet_2heads_multi)
    :param num_classes2: Number of classes for the second head in two heads model (only for unet_2heads_multi)
    :return: Model
    """
    if model_name == 'uresnet18':
        model = Unet(in_channels=channels, classes=num_classes, encoder_name='resnet18',
                     encoder_weights=encoder_weights)
    elif model_name == 'uresnet34':
        model = Unet(in_channels=channels, classes=num_classes, encoder_name='resnet34',
                     encoder_weights=encoder_weights)
    elif model_name == 'uresnet50':
        model = Unet(in_channels=channels, classes=num_classes, encoder_name='resnet50',
                     encoder_weights=encoder_weights)
    elif model_name == 'uresnet101':
        model = Unet(in_channels=channels, classes=num_classes, encoder_name='resnet101',
                     encoder_weights=encoder_weights)
    elif model_name == 'uresnet152':
        model = Unet(in_channels=channels, classes=num_classes, encoder_name='resnet152',
                     encoder_weights=encoder_weights)
    elif model_name == "dinov2":
        id2label = list(range(num_classes))
        model = dinov2_seg.Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",
                                                                         id2label=id2label, num_labels=num_classes)
        for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False
    elif model_name == "dinov2_2heads":
        id2label = list(range(num_classes))
        model = dinov2_seg_2heads.Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",
                                                                                id2label=id2label,
                                                                                num_labels=num_classes)
        for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False

    elif model_name == "unet_2heads":
        model = unet_2heads.Unet(in_channels=channels, classes=num_classes, encoder_name='resnet34',
                                 encoder_weights=encoder_weights, head_size=head_size, classes2=num_classes2)
    elif model_name == "unet_2decoders":
        model = unet_2decoders.Unet(in_channels=channels, classes=num_classes, encoder_name='resnet34',
                                    encoder_weights=encoder_weights)
    else:
        raise NotImplementedError(f' {model_name} is not implemeted...')
    if pre_trained_weight is not None:
        model.load_state_dict(torch.load(pre_trained_weight, map_location=device)['model_state_dict'])

    return model.to(device).eval()


def load_optimizer(optimizer_name, model, learning_rate, weight_decay=1e-4):
    """
    Load optimizer based on the optimizer name
    :param optimizer_name:
    :param model:
    :param learning_rate:
    :param weight_decay:
    :return: optimizer
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f' {optimizer_name} is not implemeted...')
    return optimizer
