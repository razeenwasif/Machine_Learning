from model import CNN
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy
import random
import optuna
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

emnist_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't',
    47: 'j', 48: 'k', 49: 'l', 50: 'm', 51: 'o', 52: 'p', 53: 's', 54: 'u', 55: 'v', 56: 'w',
    57: 'x', 58: 'y', 59: 'z'
}

def decode_emnist(pred_idx):
    return emnist_mapping[pred_idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
])

test_data = datasets.EMNIST(
    root='./data', split='balanced', train=False,
    download=True, transform=transform
)

# Load the model 
model = torch.load('./src/model.torch')

decoded_targets = [decode_emnist(target) for _, target in test_data]

def predict_images(attempt=100):
    pass
    model.eval()
    for i in range(attempt):
        rand_idx = random.randint(0, len(test_data) - 1)
        data, _ = test_data[rand_idx]
        data = data.unsqueeze(0).to(device)
        output = model(data)
        pred_idx = output.argmax(dim=1, keepdim=True).item()
        prediction = decode_emnist(pred_idx)
        actual = decoded_targets[rand_idx]
        print(f"Prediction: {prediction}, Actual: {actual}")
        image = data.squeeze(0).squeeze(0).cpu().numpy()
        plt.imshow(image, cmap='gray')
        plt.show()

predict_images(100)
print(device)
