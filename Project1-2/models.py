# you can import pretrained models for experimentation & add your own created models
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import torch.nn.functional as F  # function library, with relu, softmax...


class fc_model(nn.Module):

    def __init__(self, input_size, num_classes=11, dropout=0.5):
        """
            A linear model for image classification.
        """

        super(fc_model, self).__init__()
        self.cnn_layers = nn.Sequential(
            # First layer
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Second layer
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Third Layer
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Fourth Layer
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Fifth Layer
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 11),
        )

    def forward(self, x):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
        # image_vectorized --> batch_size * image_features

# =======================================
