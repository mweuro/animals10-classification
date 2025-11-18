import torch.nn as nn
import torchvision.models as models


class ResNet18Model(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Load ResNet18 model
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)

        # Modify the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def finetune(self):
        self.freeze_all()
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, x):
        return self.model(x)
