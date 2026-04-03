import torch
import torch.nn as nn
import torch.ao.quantization as tq


class NetQAT(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # Layer1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU()

        # Layer2
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Layer3
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop3 = nn.Dropout(p=0.5)

        # Layer4
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=11, stride=4, padding=2)
        self.relu4 = nn.ReLU()

        # Layer5a + Layer5b
        self.conv5a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu5b = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop5 = nn.Dropout(p=0.5)

        # Layer6
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        # Layer7
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.relu7 = nn.ReLU()

        # Layer8
        self.conv8 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.relu8 = nn.ReLU()

        # Layer9
        self.conv9 = nn.Conv2d(in_channels=384, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()

        # Layer10
        self.conv10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.relu10 = nn.ReLU()

        # Layer11
        self.conv11 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.relu11 = nn.ReLU()

        # FC: 40 -> 100 -> 100 -> 100 -> 10
        self.fc1 = nn.Linear(in_features=40, out_features=100)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        self.relu_fc3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = self.quant(x)

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.relu4(self.conv4(x))

        x = self.relu5a(self.conv5a(x))
        x = self.relu5b(self.conv5b(x))
        x = self.pool5(x)
        x = self.drop5(x)

        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.relu9(self.conv9(x))
        x = self.relu10(self.conv10(x))
        x = self.relu11(self.conv11(x))

        x = torch.flatten(x, 1)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))
        x = self.fc4(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        tq.fuse_modules(self, [
            ['conv1', 'relu1'],
            ['conv2', 'relu2'],
            ['conv3', 'relu3'],
            ['conv4', 'relu4'],
            ['conv5a', 'relu5a'],
            ['conv5b', 'relu5b'],
            ['conv6', 'relu6'],
            ['conv7', 'relu7'],
            ['conv8', 'relu8'],
            ['conv9', 'relu9'],
            ['conv10', 'relu10'],
            ['conv11', 'relu11'],
            ['fc1', 'relu_fc1'],
            ['fc2', 'relu_fc2'],
            ['fc3', 'relu_fc3'],
        ], inplace=True)
