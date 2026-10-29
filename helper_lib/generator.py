import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device="cpu", num_samples=16):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        fake_images = model(noise)

    fake_images = (fake_images.cpu() + 1) / 2

    grid = make_grid(
        fake_images,
        nrow=4,
        padding=2
    )

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("GAN Generated MNIST Digits")
    plt.savefig("outputs/generated_digits.png", bbox_inches="tight")
    plt.show()
