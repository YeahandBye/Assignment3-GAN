import io
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from torchvision.utils import make_grid
from PIL import Image

from helper_lib.model import Generator
from helper_lib.utils import load_model

app = FastAPI(title="MNIST GAN API")

device = "cpu"

generator = load_model(
    Generator(),
    "checkpoints/generator.pth",
    device=device
)


@app.get("/")
def home():
    return {"message": "MNIST GAN API is running!"}


@app.get("/generate")
def generate(num_samples: int = 16):
    if not 1 <= num_samples <= 64:
        return {"error": "num_samples must be between 1 and 64"}

    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        images = generator(noise)

    images = (images + 1) / 2

    grid = make_grid(images, nrow=4, padding=2)
    grid = (grid * 255).clamp(0, 255).byte()

    image = Image.fromarray(
        grid.permute(1, 2, 0).cpu().numpy()
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
