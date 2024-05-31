import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_classes=2, avg_pool=False):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.avg_pool = avg_pool
        self.width = tokenW
        self.height = tokenH
        self.num_labels = num_classes
        self.upsample = nn.Upsample(size=(448, 448), mode='nearest')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, num_classes, (1, 1))

    def forward(self, embeddings, mask_batch):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        x = self.upsample(embeddings)
        x = x * mask_batch
        x = self.pool(x)
        if self.avg_pool:
            return x
        else:
            return self.conv(x)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config, num_classes, avg_pool):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 32, 32, num_classes=num_classes,
                                           avg_pool=avg_pool)

    def forward(self, pixel_values, mask, output_hidden_states=False, output_attentions=False, labels=None):
        # use frozen features
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings, mask, )
        # logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear",
        #                                          align_corners=False)
        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.squeeze(), labels)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
