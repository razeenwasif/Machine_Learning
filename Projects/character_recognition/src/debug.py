import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load your TorchScript model
model = torch.jit.load('../model.pth')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emnist_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't',
    47: 'j', 48: 'k', 49: 'l', 50: 'm', 51: 'o', 52: 'p', 53: 's', 54: 'u', 55: 'v', 56: 'w',
    57: 'x', 58: 'y', 59: 'z'
}

transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_png(png_path, model, device):
    # Open and resize to 28x28
    img_pil = Image.open(png_path).convert('L')
    img_pil = img_pil.resize((28, 28))

    # Apply the same transform used in training
    img_tensor = transform(img_pil)

    # Optional: visualize the final tensor for debugging
    plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')
    plt.title(f"Model Input: {os.path.basename(png_path)}")
    plt.show()

    # Run inference
    img_tensor = img_tensor.unsqueeze(0).to(device)
    output = model(img_tensor)
    probs = torch.exp(output)
    pred_idx = probs.argmax(dim=1).item()

    predicted_char = emnist_mapping[pred_idx]
    print(f"File: {png_path} | Predicted index: {pred_idx}, Mapped character: {predicted_char}")
    return predicted_char

# ------------------------------------------------------------------
# Main part: Loop through your test_imgs folder
folder_path = "./test_imgs"
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        full_path = os.path.join(folder_path, filename)
        predict_png(full_path, model, device)

