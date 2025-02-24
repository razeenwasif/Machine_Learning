import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loader
from model import CNN
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word list
with open("./wordlist/words_dictionary.json", "r") as f:
    WORD_LIST = json.load(f)

WORD_LIST = list(WORD_LIST.keys())

# filter out empty strings
WORD_LIST = [word for word in WORD_LIST if word.strip()]

# Training Function
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Training Setup
if __name__ == "__main__":
    train_loader = get_data_loader("./synthetic_dataset/", WORD_LIST, batch_size=32, shuffle=True)
    model = CNN(dropout_rate=0.5, fc1_neurons=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, criterion, epochs=10)
    torch.save(model.state_dict(), "./model.pth")
    print("Model saved.")


