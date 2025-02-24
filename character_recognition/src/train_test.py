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

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomRotation(10),  # Random rotations up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Random translate
])

transform_test = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
])

# Load the dataset
train_data = datasets.EMNIST(
    root='./data', split='balanced', train=True,
    download=True, transform=transform_train
)
    
test_data = datasets.EMNIST(
    root='./data', split='balanced', train=False,
    download=True, transform=transform_test
)

num_workers = multiprocessing.cpu_count()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scaler = GradScaler()

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # collect preds and true labels 
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    # calculate accuracy
    accuracy = 100. * correct / len(test_loader.dataset)

    # other metrics
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=1)

    print(f"Test Loss: {test_loss / len(test_loader.dataset):.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}") 
    
    return test_loss / len(test_loader.dataset), accuracy


# ------------------------- Hyperparameter tuning -------------------------------------#
# Objective function for Optuna
def objective(trial):
    # Suggest values for hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 1.0)
    fc1_neurons = trial.suggest_int('fc1_neurons', 50, 200)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(dropout_rate, fc1_neurons).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # early stopping parameters 
    best_accuracy = 0 
    # number of epochs to wait before stopping 
    patience = 5 
    epochs_without_improvement = 0 

    # Training loop
    for epoch in range(1, 21): #21
        train(model, device, train_loader, optimizer, criterion)
        test_loss, accuracy = test(model, device, test_loader, criterion)

        # early stopping condition
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0 
        else: 
            epochs_without_improvement += 1 

        if epochs_without_improvement > patience:
            print("stopped early")
            break
    
    # Optuna minimizes the returned value, so we return -accuracy to maximize it
    return -accuracy
# ------------------------------End of Hyperparameter tuning-------------------------------#

# Training Setup
if __name__ == "__main__":
    # Create a study to optimize the objective
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=80)  # Run 50 trials of optimization

    # Get the best trial and hyperparameters
    print("Best trial:")
    trial = study.best_trial

    print(f"best trial value: {trial.value}")
    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    #training and evalulation after optimization
    best_params = trial.params

    # Create data loaders
    loaders = {
            'train': DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True, num_workers=num_workers),
            'test': DataLoader(test_data, batch_size=best_params["batch_size"], shuffle=True, num_workers=num_workers),
            }

    model = CNN(best_params["dropout_rate"], best_params["fc1_neurons"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, 36): #26
        train(model, device, loaders['train'], optimizer, criterion)
        test_loss, accuracy = test(model, device, loaders['test'], criterion)
        scheduler.step()

        print(f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")


    # final model evaluation
    y_true = []
    y_pred = []

    for data, target in loaders['test']:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Save the model 
    torch.save(model.state_dict(), './weights_and_biases.torch')
    torch.save(model, './model.torch')

    # save model for deployment
    scripted_model = torch.jit.script(model)
    scripted_model.save("model.pth")
    print("Model saved.")

