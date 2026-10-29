import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def get_model(model_name):
    if model_name == "GAN":
        return Generator(), Discriminator()

    raise ValueError(f"Unknown model: {model_name}")
