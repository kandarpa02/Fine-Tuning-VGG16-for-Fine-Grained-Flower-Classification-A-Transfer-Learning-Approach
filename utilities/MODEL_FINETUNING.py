import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vgg16

## Hyperparameters
num_classes = 102
Freeze_param_index = 9 #Freezes parameters, making the model train last few parameters



# VGG_16 was trained on ImageNet dataset, we have to finetune the model according to our usecase
def FINETUNE_VGG():
    vgg = vgg16(pretrained = True)
    vgg.classifier[6] = nn.Linear(4096, num_classes)

    for param in list(vgg.features.parameters())[:Freeze_param_index]:
        param.requires_grad = False

    vgg.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

    return vgg 

if __name__ == "__main__":
    FINETUNE_VGG()