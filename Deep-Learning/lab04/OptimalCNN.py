import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class OptimalCNN(nn.Module):
    """An optimized CNN for CIFAR-10 featuring Batch Normalization,
    Global Average Pooling, and a deeper VGG-style architecture."""
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 8x8 -> 4x4 
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Avg Pooling 
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), # Stronger regularization for deeper network 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

def train_optimal_model(model, loader, optimizer, criterion, scheduler, epochs, device, val_loader=None):
    """Custom training loop with OneCycleLR scheduler support (updates every batch)"""
    train_losses, val_losses = [], [] 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step() # Critical: OneCycleLR updates every batch
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(loader))
        
        if val_loader:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for d, t in val_loader:
                    d, t = d.to(device), t.to(device)
                    v_loss += criterion(model(d), t).item()
            val_losses.append(v_loss / len(val_loader))
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}')
            
    return train_losses, val_losses

if __name__ == "__main__":
    # Enhanced Data Augmentation 
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # reload dataset with augmentation 
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=aug_transform)
    opt_train_loader = DataLoader(full_train, batch_size=128, shuffle=True, num_workers=2) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt_model = OptimalCNN().to(device)
    opt_optimizer = torch.optim.AdamW(opt_model.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_criterion = nn.CrossEntropyLoss() 

    opt_epochs = 20 
    opt_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt_optimizer, max_lr=0.01, steps_per_epoch=len(opt_train_loader), epochs=opt_epochs
    )

    print(f"Starting Model Training")
    train_optimal_model(opt_model, opt_train_loader, opt_optimizer, opt_criterion, opt_scheduler, opt_epochs, device)
