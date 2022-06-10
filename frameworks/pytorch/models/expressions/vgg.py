
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_features(nn.Module):

    def _make_layers(self, in_channels, cfg, batch_norm=True):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def __init__(self, in_channels, cfg, batch_norm=True):
        super(VGG_features, self).__init__()
        self.features = self._make_layers(in_channels, cfg, batch_norm=batch_norm)
    
    def forward(self, x):
        if len(x.shape) == 3: # (B, H, W)
            x.unsqueeze_(1) # (B, C, H, W)
        x = self.features(x)
        x1, _ = x.max(dim=-1)
        x2 = x.mean(dim=-1)
        x = torch.cat((x1, x2), dim=-1)
        x = x.flatten(1)
        return x


class VGG(nn.Module):
    # VGG is a Network with a fetures layer and a classifier layer
    # It uses 3x3 and 1x1 convolutions and max poolings at the features layer and three linears at the classifier layer. All the activation functions are defined as ReLU

    def __init__(self, num_classes, features_extractor, classifier_in_size=1024, classifier_hidden=64, classifier_dropout=0.4):
        # classifier_in_size = num_features_channels * width_after_convolutions * height_after_convolutions
        super(VGG, self).__init__()
        self.features = features_extractor
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_in_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x # (B, num_classes) # if x.shape[-1] == 1 : x.squeeze(-1)
