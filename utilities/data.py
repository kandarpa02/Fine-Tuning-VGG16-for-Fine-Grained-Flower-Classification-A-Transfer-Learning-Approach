import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102


class DATA_PREPERATION:

    @staticmethod
    def DOWNLOAD_DATA():
        return Flowers102(root="data/flowers-102", download=True)
    
    @staticmethod
    def PREPROCESS():
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )])
        
        train_dataset = Flowers102(root="data/flowers-102", split="train", download=False, transform=transform)
        val_dataset = Flowers102(root="data/flowers-102", split="val", download=False, transform=transform)
        test_dataset = Flowers102(root="data/flowers-102", split="test", download=False, transform=transform)

        return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    DATA_PREPERATION()

