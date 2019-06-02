import torch
import torch.nn as nn
from bayesian.ConcreteDropoutLinear import ConcreteDropoutConvolutional, ConcreteDropoutLinear


class CNN(nn.Module):
    """CNN."""

    def __init__(self, weight_regularizer, dropout_regularizer, **kwargs):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_drop1 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)
        self.conv_drop2 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)
        self.conv_drop3 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)
        self.conv_drop4 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)
        self.conv_drop5 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)
        self.conv_drop6 = ConcreteDropoutConvolutional(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer, temp = 2. / 3.)

        self.conc_drop1 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropoutLinear(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        # Conv Layer block 1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True)
        )

        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4096, 1024),
            # nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        regularization = torch.empty(9, device=x.device)

        # conv layers
        # x = self.conv_layer(x)
        x, reg[0] = self.conv_drop1(x, self.conv_layer1)
        x, reg[1] = self.conv_drop2(x, self.conv_layer2)
        x, reg[2] = self.conv_drop3(x, self.conv_layer3)
        x, reg[3] = self.conv_drop4(x, self.conv_layer4)
        x, reg[4] = self.conv_drop5(x, self.conv_layer5)
        x, reg[5] = self.conv_drop6(x, self.conv_layer6)

        x = x.view(x.size(0), -1)

        x, reg[6] = self.conc_drop1(x, self.fc1)
        x, reg[7] = self.conc_drop2(x, self.fc2)
        x, reg[8] = self.conc_drop3(x, self.fc3)
        
        return x