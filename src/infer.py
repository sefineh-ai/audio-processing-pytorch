import torch
import numpy as np
from train import SimpleAudioClassifier

def load_model(model_path, num_features=13, seq_len=50, num_classes=10):
    model = SimpleAudioClassifier(num_features, seq_len, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, features):
    with torch.no_grad():
        x = torch.tensor(features).unsqueeze(0)  # Add batch dim
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return pred

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Inference")
    parser.add_argument('--model', type=str, default="../models/simple_audio_classifier.pth", help='Path to model')
    parser.add_argument('--features', type=str, required=True, help='Path to .npy features file')
    args = parser.parse_args()

    features = np.load(args.features)
    model = load_model(args.model, num_features=features.shape[0], seq_len=features.shape[1])
    pred = predict(model, features)
    print(f"Predicted class: {pred}") 