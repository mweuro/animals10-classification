import torch.nn as nn
import torchvision.models as models


class ResNet18Model(nn.Module):
    """
    ResNet18 model for image classification.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initializes the ResNet18 model.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use a pretrained model. Default is True.
        """
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
        """
        Freezes all layers in the model.

        Returns:
            None
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreezes all layers in the model.

        Returns:
            None
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def finetune(self):
        """
        Freezes all layers except the last block and the fully connected layer for fine-tuning.

        Returns:
            None
        """
        self.freeze_all()
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    def get_trainable_params(self):
        """
        Returns a list of parameters that are trainable (i.e., require gradients).

        Returns:
            list: List of trainable parameters.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)
