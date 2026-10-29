# Assignment 3 - Generative Adversarial Networks (GAN)

## Project Overview

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate handwritten digits from the MNIST dataset.

The assignment includes GAN architecture implementation, model training, image generation, and API integration.

## GAN Architecture

### Generator
- Input: 100-dimensional random noise
- Fully connected layer: 7 x 7 x 128
- ConvTranspose2D: 128 to 64, followed by BatchNorm and ReLU
- ConvTranspose2D: 64 to 1, followed by Tanh
- Output: 1 x 28 x 28 grayscale image

### Discriminator
- Input: 1 x 28 x 28 grayscale image
- Conv2D: 1 to 64, followed by LeakyReLU
- Conv2D: 64 to 128, followed by BatchNorm and LeakyReLU
- Flatten and fully connected layer
- Sigmoid output for real/fake probability

## Dataset

The model is trained using the MNIST handwritten digit dataset.

Images are normalized to the range [-1, 1].

## Training

The GAN was trained for 5 epochs using:
- Binary Cross Entropy Loss
- Adam optimizer
- Learning rate: 0.0002
- Batch size: 64

Run training:

python3 train_gan.py

## Image Generation

The trained Generator produces handwritten digit images.

Generated samples are saved in:

outputs/generated_digits.png

## API

The project includes a FastAPI application for generating handwritten digits.

Start the API:

python3 -m uvicorn api:app --reload

Available endpoints:

- GET /
- GET /generate?num_samples=16

## Project Structure

- helper_lib/model.py: GAN models
- helper_lib/trainer.py: GAN training
- helper_lib/data_loader.py: MNIST data loading
- helper_lib/generator.py: Image generation
- helper_lib/utils.py: Model saving and loading
- train_gan.py: Training script
- api.py: FastAPI application

## Results

The GAN successfully generated MNIST-style handwritten digits after training.
