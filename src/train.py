import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from audio_dataset import AudioFeatureDataset
import os
import argparse

class SimpleAudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

def train(model, loader, criterion, optimizer, device):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio classifier on extracted features")
    parser.add_argument('--features_root', type=str, required=True, help='Path to root directory of extracted features')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_out', type=str, default="../models/simple_audio_classifier.pth")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AudioFeatureDataset(args.features_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Infer input_dim and num_classes
    sample_X, _ = dataset[0]
    input_dim = int(np.prod(sample_X.shape))
    num_classes = len(dataset.class_to_idx)

    model = SimpleAudioClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss = train(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"Model saved to {args.model_out}") 