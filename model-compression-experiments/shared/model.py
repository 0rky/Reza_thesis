import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)

        # Layer2
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)

        # Layer3
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop3 = nn.Dropout(p=0.5)

        # Layer4
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=11, stride=4, padding=2)

        # Layer5a + Layer5b
        self.conv5a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv5b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop5 = nn.Dropout(p=0.5)

        # Layer6
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        # Layer7
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0)

        # Layer8
        self.conv8 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0)

        # Layer9
        self.conv9 = nn.Conv2d(in_channels=384, out_channels=10, kernel_size=3, stride=1, padding=1)

        # Layer10
        self.conv10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, stride=1, padding=0)

        # Layer11
        self.conv11 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, stride=1, padding=0)

        # FC
        self.fc1 = nn.Linear(in_features=40, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)
        x = self.drop5(x)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x