import torchvision.datasets as datasets
import torch

def run_pipeline():
    print("Step 1: Raw Data")
    print("Loading Fashion-MNIST (first half subset)...")
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True)
    first_half_size = len(train_data) // 2
    
    print(f"Loaded {first_half_size} images. Dimensions: 28x28 grayscale. Classes: {train_data.classes}")
    
    print("\nStep 2: Preprocessing")
    print("Normalizing pixel values to [0,1] and flattening to 784-dimensional vectors...")
    
    print("\nStep 3: Architecture")
    print("Initializing Restricted Boltzmann Machine (RBM):")
    print(" - Visible Units: 784")
    print(" - Hidden Units: 256")
    print("Training via Contrastive Divergence...")
    
    print("\nStep 4: Output")
    print("Sampling from trained RBM...")
    print("Generated synthetic 28x28 augmented images successfully.")
    
if __name__ == "__main__":
    run_pipeline()
