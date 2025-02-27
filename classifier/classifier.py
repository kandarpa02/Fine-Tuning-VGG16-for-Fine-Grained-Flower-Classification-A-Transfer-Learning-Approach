import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from utilities.data import DATA_PREPERATION
from utilities.MODEL_FINETUNING import FINETUNE_VGG


FLOWER_102_CLASSES = {
    1: "Pink Primrose",
    2: "Hard-leaved Pocket Orchid",
    3: "Canary Creeper",
    4: "Sweet Pea",
    5: "English Marigold",
    6: "Tiger Lily",
    7: "Moon Orchid",
    8: "Bird of Paradise",
    9: "Monkshood",
    10: "Globe Thistle",
    11: "Snapdragon",
    12: "Colt’s Foot",
    13: "King Protea",
    14: "Skeleton Weed",
    15: "Yellow Iris",
    16: "Globe-flower",
    17: "Purple Coneflower",
    18: "Peruvian Lily",
    19: "Balloon Flower",
    20: "Giant White Arum Lily",
    21: "Fire Lily",
    22: "Pincushion Flower",
    23: "Fritillary",
    24: "Red Ginger",
    25: "Grapevine",
    26: "Corn Poppy",
    27: "Prince of Wales Feathers",
    28: "Stemless Gentian",
    29: "Artichoke",
    30: "Sweet William",
    31: "Carnation",
    32: "Garden Phlox",
    33: "Love in the Mist",
    34: "Mexican Aster",
    35: "Alpine Sea Holly",
    36: "Ruby-lipped Cattleya",
    37: "Cape Flower",
    38: "Great Masterwort",
    39: "Siam Tulip",
    40: "Lenten Rose",
    41: "Barbeton Daisy",
    42: "Dahlia",
    43: "Sword Lily",
    44: "Poinsettia",
    45: "Bolero Deep Blue",
    46: "Wallflower",
    47: "Marigold",
    48: "Buttercup",
    49: "Oxeye Daisy",
    50: "Common Dandelion",
    51: "Petunia",
    52: "Wild Pansy",
    53: "Primula",
    54: "Sunflower",
    55: "Pelargonium",
    56: "Bishop of Llandaff",
    57: "Gaillardia",
    58: "Gazania",
    59: "Azalea",
    60: "Water Lily",
    61: "Rose",
    62: "Thorn Apple",
    63: "Morning Glory",
    64: "Passion Flower",
    65: "Lotus",
    66: "Columbine",
    67: "Bouvardia",
    68: "Tree Poppy",
    69: "Magnoliids",
    70: "Cyclamen",
    71: "Watercress",
    72: "Canna Lily",
    73: "Hippeastrum",
    74: "Bee Balm",
    75: "Ball Moss",
    76: "Foxglove",
    77: "Bougainvillea",
    78: "Camellia",
    79: "Mallow",
    80: "Mexican Petunia",
    81: "Bromelia",
    82: "Blanket Flower",
    83: "Trumpet Creeper",
    84: "Black-eyed Susan",
    85: "Silverbush",
    86: "Californian Poppy",
    87: "Osteospermum",
    88: "Clematis",
    89: "Hibiscus",
    90: "Echinacea",
    91: "Fireweed",
    92: "Common Columbine",
    93: "Desert Rose",
    94: "Tree Mallow",
    95: "Magnolia",
    96: "Bishop’s Weed",
    97: "Gaura",
    98: "Geranium",
    99: "Orange Dahlia",
    100: "Pansy",
    101: "Butter Daisy",
    102: "Daffodil",
}

# Initialize model and device
model = FINETUNE_VGG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
model_path = os.path.join(project_root, 'Fine-Tuning-VGG16-for-Fine-Grained-Flower-Classification-A-Transfer-Learning-Approach/classifier', 'model', 'model.pth')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    print("CUDA is not available. Loading model on CPU.")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model = model.to(device)


# Prepare test data loader
test_data = DATA_PREPERATION().PREPROCESS()[2]
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

def prediction(images):
    with torch.no_grad():
        output = model(images)
    prob = F.softmax(output, dim=1)
    predicted_classes = torch.argmax(prob, dim=1)
    predicted_labels = [FLOWER_102_CLASSES[idx] for idx in predicted_classes.cpu().numpy()]
    return predicted_labels

# Run prediction and plot results for a specific batch index
def classify(batch_index):
    for i, (images, labels) in enumerate(itertools.islice(test_loader, batch_index, batch_index+1)):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = prediction(images)

        true_labels = [FLOWER_102_CLASSES[idx] for idx in labels.cpu().numpy()]
        predicted_labels = predictions

        images = images.cpu().numpy().transpose((0, 2, 3, 1))
        images = np.clip(images, 0, 1)

        batch_size = len(images)
        cols = min(6, batch_size)
        rows = (batch_size + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
        axes = axes.flatten()

        for j, ax in enumerate(axes[:batch_size]):
            ax.imshow(images[j])
            ax.axis("off")
            color = "green" if true_labels[j] == predicted_labels[j] else "red"
            ax.set_title(f"True: {true_labels[j]}\nPred: {predicted_labels[j]}", fontsize=10, color=color)

        # Remove empty axes
        for ax in axes[batch_size:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    x = sys.argv[1]
    classify(int(x))