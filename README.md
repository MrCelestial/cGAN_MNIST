![generated_images](assets/generated_images)
# Conditional GAN for MNIST Digit Generation

This project implements a Conditional Generative Adversarial Network (cGAN) for generating handwritten digits based on the MNIST dataset. The implementation includes interactive components for exploring the model's capabilities directly in Google Colab.

## Overview

This Conditional GAN allows for the generation of specific digit classes by conditioning both the generator and discriminator on class labels. The project demonstrates how to:

1. Build and train a conditional GAN architecture
2. Generate specific digit classes on demand
3. Visualize and interact with the generation process
4. Create smooth transitions between different digit classes

## Features

- **Conditional Digit Generation**: Generate specific digits (0-9) by conditioning the model
- **Interactive Interface**: Jupyter widgets for real-time digit generation and visualization
- **Latent Space Exploration**: Explore the latent space by manipulating noise vectors
- **Digit Morphing**: Create smooth transitions between different digit classes
- **Batch Generation**: Generate multiple samples at once to observe variations

## Implementation Details

### Model Architecture

#### Generator
- Input: Random noise vector (z) concatenated with one-hot encoded class label
- Architecture: Multiple transposed convolutional layers with batch normalization
- Output: 28x28 grayscale image of the specified digit class

#### Discriminator
- Input: Image concatenated with class label (expanded as additional channels)
- Architecture: Convolutional layers with leaky ReLU activations
- Output: Probability that the input image is real or fake

### Training Process
- Adversarial training with binary cross entropy loss
- Adam optimizer with learning rate of 0.0002 and betas=(0.5, 0.999)
- Training for 20 epochs on the MNIST dataset

## How to Use

1. Open the [Colab Notebook](https://colab.research.google.com/drive/1Nqt_IdtW7Zz7SRMMjb_LBTtRs2QEvUrs)
2. Run all cells to train the model (or load pre-trained weights)
3. Use the interactive widgets to:
   - Generate specific digit classes
   - Create batches of samples
   - Explore variations within a digit class
   - Create morphing animations between digits

## Interactive Components

The notebook includes several interactive widgets:
- Digit class selector
- Latent vector manipulator
- Batch size controller
- Animation generator for transitions between digits

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- ipywidgets
- imageio

## Results

The model successfully generates realistic handwritten digits for all classes (0-9). The conditional aspect allows for precise control over which digit class to generate, while the latent space encoding captures various style aspects of handwriting such as slant, thickness, and shape variations.

## Acknowledgments

- MNIST Dataset
- Original cGAN paper: "Conditional Generative Adversarial Nets" by Mehdi Mirza and Simon Osindero
