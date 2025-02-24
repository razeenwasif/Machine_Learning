import torch
from dataset import get_data_loader
from model import CNN
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

# Load Model and Test
if __name__ == "__main__":
    test_loader = get_data_loader("./data/synthetic_dataset/", batch_size=32, shuffle=False)
    model = CNN(dropout_rate=0.5, fc1_neurons=128).to(device)
    model.load_state_dict(torch.load("./model.pth"))
    test(model, test_loader)

