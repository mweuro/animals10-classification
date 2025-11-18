from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_ds = datasets.ImageFolder("data/train", transform=train_transforms)
val_ds   = datasets.ImageFolder("data/val",   transform=test_transforms)
test_ds  = datasets.ImageFolder("data/test",  transform=test_transforms)

train_loader = DataLoader(train_ds, shuffle=True, **config['DATALOADER'])
val_loader   = DataLoader(val_ds, shuffle=False, **config['DATALOADER'])
test_loader  = DataLoader(test_ds, shuffle=False, **config['DATALOADER'])