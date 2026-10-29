import os
import torch

def save_model(model, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device="cpu"):
    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
