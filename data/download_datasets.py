"""
Download and prepare all datasets for the project.

This script downloads:
- MNIST (for LeNet-5)
- CIFAR-10 (for ResNet-18)
- IMDB Reviews (for LSTM sentiment classifier)
- WikiText-2 (for GRU language model)
"""

import os
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import nltk


def download_mnist(data_dir: str = "./data"):
    """Download MNIST dataset."""
    print("Downloading MNIST...")
    mnist_dir = os.path.join(data_dir, "mnist")
    os.makedirs(mnist_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download training and test sets
    torchvision.datasets.MNIST(
        root=mnist_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    torchvision.datasets.MNIST(
        root=mnist_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✓ MNIST downloaded to {mnist_dir}")


def download_cifar10(data_dir: str = "./data"):
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10...")
    cifar_dir = os.path.join(data_dir, "cifar10")
    os.makedirs(cifar_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download training and test sets
    torchvision.datasets.CIFAR10(
        root=cifar_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    torchvision.datasets.CIFAR10(
        root=cifar_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✓ CIFAR-10 downloaded to {cifar_dir}")


def download_imdb(data_dir: str = "./data"):
    """Download IMDB Reviews dataset."""
    print("Downloading IMDB Reviews...")
    imdb_dir = os.path.join(data_dir, "imdb")
    os.makedirs(imdb_dir, exist_ok=True)
    
    # Download using HuggingFace datasets
    dataset = load_dataset("imdb", cache_dir=imdb_dir)
    
    print(f"✓ IMDB Reviews downloaded to {imdb_dir}")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples: {len(dataset['test'])}")


def download_wikitext2(data_dir: str = "./data"):
    """Download WikiText-2 dataset."""
    print("Downloading WikiText-2...")
    wikitext_dir = os.path.join(data_dir, "wikitext")
    os.makedirs(wikitext_dir, exist_ok=True)
    
    try:
        # Download WikiText-2 dataset for language modeling
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=wikitext_dir)
        print(f"✓ WikiText-2 downloaded to {wikitext_dir}")
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(dataset['validation'])}")
        print(f"  Test samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"❌ WikiText-2 download failed: {e}")
        print("  You may need to manually download the dataset")
        raise


def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"Warning: NLTK download failed: {e}")


def main():
    """Download all datasets."""
    print("=" * 60)
    print("AheadOfItsTime - Dataset Download")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    print(f"\nData will be saved to: {data_dir.absolute()}\n")
    
    # Download all datasets
    try:
        download_mnist(str(data_dir))
        print()
        
        download_cifar10(str(data_dir))
        print()
        
        download_imdb(str(data_dir))
        print()
        
        download_wikitext2(str(data_dir))
        print()
        
        download_nltk_data()
        print()
        
        print("=" * 60)
        print("✓ All datasets downloaded successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error downloading datasets: {e}")
        print("Please check your internet connection and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
