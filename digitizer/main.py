import torch
import matplotlib.pyplot as plt
from dataset import transform
from model import CNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(image_path, model):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(dim=1, keepdim=True).item()
    
    print(f"Prediction: {pred_idx}")
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.show()

if __name__ == "__main__":
    model = CNN(dropout_rate=0.5, fc1_neurons=128).to(device)
    model.load_state_dict(torch.load("./model.pth"))
    predict("./data/synthetic_dataset/sample.png", model)

