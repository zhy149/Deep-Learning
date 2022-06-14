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
        # print("here", input_size, num_classes)
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1)  # 5 or 3
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)  # error place
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  # new add
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(12544, 360)
        self.fc2 = nn.Linear(360, 180)
        self.fc3 = nn.Linear(180, 64)
        self.fc4 = nn.Linear(64, 11)

        # initialize parameters (write code to initialize parameters here)

    def forward(self, x):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """
        # print("x layer shape", x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # print("this is the shape", x.shape)
        x = x.view(-1, 12544)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # image_vectorized --> batch_size * image_features

        return x

# =======================================
