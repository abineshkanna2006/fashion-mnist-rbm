# Fashion-MNIST Image Augmentation with Restricted Boltzmann Machines

This repository contains a demonstration of using a Restricted Boltzmann Machine (RBM) for image augmentation on the Fashion-MNIST dataset.

## Dataset
- **Name:** Fashion-MNIST
- **Variant:** First Half Slice (30,000 images)
- **Link:** [Fashion-MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## Pipeline Outline
1. **Raw Data:** Fashion-MNIST first-half subset (30,000 grayscale 28x28 images, pixel values 0-255).
2. **Preprocessing:** Normalize pixel values to [0,1], flatten each 28x28 matrix into a 784-dimensional vector.
3. **Architecture:** Restricted Boltzmann Machine (784 visible units, 256 hidden units), trained via Contrastive Divergence.
4. **Output:** Synthetically augmented 784-dimensional vectors sampled from the RBM, reshaped to 28x28 matrices as new training data.

## Demo
Run `demo.py` to see a placeholder of the augmentation pipeline in action.
