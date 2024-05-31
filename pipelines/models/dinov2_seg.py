import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//2),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//4),
            nn.Conv2d(in_channels//4, in_channels//4, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//4),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels//4, in_channels//6, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//6),
            nn.Conv2d(in_channels//6, in_channels//6, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//6),

            nn.Conv2d(in_channels//6, num_labels, (1, 1)),
        )

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False):
        # use frozen features

        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear",
                                                 align_corners=False)

        # loss = None
        # if labels is not None:
        #     # important: we're going to use 0 here as ignore index instead of the default -100
        #     # as we don't want the model to learn to predict background
        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  #
        #     # print('Logits', logits.shape, 'Labels', labels.shape)
        #     loss = loss_fct(logits, labels.squeeze())
        #
        # return SemanticSegmenterOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return logits
