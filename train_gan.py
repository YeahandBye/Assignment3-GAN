import os
import torch

from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_gan


def main():
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}", flush=True)

    os.makedirs("checkpoints", exist_ok=True)

    data_loader = get_data_loader(
        data_dir="./data",
        batch_size=64,
        train=True
    )

    model = get_model("GAN")

    generator, discriminator = train_gan(
        model=model,
        data_loader=data_loader,
        device=device,
        epochs=5
    )

    torch.save(
        generator.cpu().state_dict(),
        "checkpoints/generator.pth"
    )

    torch.save(
        discriminator.cpu().state_dict(),
        "checkpoints/discriminator.pth"
    )

    print("Training completed!")
    print("Models saved in checkpoints/")


if __name__ == "__main__":
    main()
