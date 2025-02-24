import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Custom Dataset Class
class HandwritingDataset(Dataset):
    def __init__(self, root_dir, word_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.labels = [f.split("_")[0] for f in self.image_files]  # Extract word from filename
        
        # Encode labels using provided word list
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(word_list)
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("L")
        label = self.encoded_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to create DataLoader
def get_data_loader(data_dir, word_list, batch_size=32, shuffle=True, num_workers=4):
    dataset = HandwritingDataset(data_dir, word_list, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    from synthetic_dataset_generator import WORD_LIST  # Import from the correct script
    dataloader = get_data_loader("./synthetic_dataset/", WORD_LIST)
    print(f"Dataset loaded: {len(dataloader.dataset)} samples")

