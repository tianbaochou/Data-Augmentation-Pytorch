import torch.nn as nn
from torchvision import models

class CNNNet(nn.Module):
    def __init__(self, model_name, nclasses=10, pretrained=False):
        super(CNNNet, self).__init__()
        if model_name == 'base-model':
            self.features = nn.Sequential(
                    nn.Conv2d(3, 6, 5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(6, 16, 5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    )
            self.classifier = nn.Sequential(
                    nn.Linear(16* 5* 5, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(128, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(128, nclasses),
                    )
            self.model_name = 'base-model'
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, nclasses),
            )
            self.model_name = 'alexnet'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'base-model':
            f = f.view(-1, 16 * 5 * 5)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        y = self.classifier(f)
        return y
