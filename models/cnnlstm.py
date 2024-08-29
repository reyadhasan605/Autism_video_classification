import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, densenet201


class CNNLSTM(nn.Module):

    def __init__(self, num_classes=7):
        super(CNNLSTM, self).__init__()
         # Initialize DenseNet-201 with pretrained weights
        self.densenet = densenet201(pretrained=True)

        # Replace DenseNet's classifier with a new fully connected layer
        self.densenet.classifier = nn.Linear(1920, 300)

        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)  # LSTM with dropout
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2)  # Dropout before the first fully connected layer
        )
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.densenet(x_3d[:, t, :, :, :])  # Pass each frame through DenseNet
            out, hidden = self.lstm(x.unsqueeze(0), hidden)  # Pass the output through LSTM

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

    # def __init__(self, num_classes=2):
    #     super(CNNLSTM, self).__init__()
    #     self.resnet = resnet101(pretrained=True)
    #     self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
    #     self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
    #     self.fc1 = nn.Linear(256, 128)
    #     self.fc2 = nn.Linear(128, num_classes)
    #
    # def forward(self, x_3d):
    #     hidden = None
    #     for t in range(x_3d.size(1)):
    #         with torch.no_grad():
    #             x = self.resnet(x_3d[:, t, :, :, :])
    #         out, hidden = self.lstm(x.unsqueeze(0), hidden)
    #
    #     x = self.fc1(out[-1, :, :])
    #     x = F.relu(x)
    #     x = self.fc2(x)
    #     return x