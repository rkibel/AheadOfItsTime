"""
Training script for LSTM Sentiment Classifier on IMDB dataset.

"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.rnn.lstm_sentiment import LSTMSentimentClassifier
from models.utils import count_parameters, save_checkpoint


class IMDBDataset(Dataset):
    """IMDB dataset wrapper."""
    
    def __init__(self, texts, labels, vocab, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize (simple whitespace tokenization)
        tokens = text.lower().split()[:self.max_length]
        
        # Convert to indices
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_vocab(texts, min_freq=5, max_vocab_size=25000):
    """Build vocabulary from texts."""
    counter = Counter()
    for text in tqdm(texts, desc="Building vocabulary"):
        tokens = text.lower().split()
        counter.update(tokens)
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add most common tokens
    for token, freq in counter.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def collate_fn(batch):
    """Collate function for batching variable-length sequences."""
    texts, labels = zip(*batch)
    
    # Get lengths before padding
    lengths = torch.tensor([len(text) for text in texts])
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels, lengths


def get_data_loaders(batch_size: int = 64, data_dir: str = "./data/imdb", max_length: int = 512):
    """
    Create IMDB data loaders.
    
    Args:
        batch_size: Batch size for training and testing
        data_dir: Directory containing IMDB data
        max_length: Maximum sequence length
        
    Returns:
        train_loader, test_loader, vocab_size
    """
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", cache_dir=data_dir)
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Build vocabulary from training set
    vocab = build_vocab(train_texts)
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_length)
    
    # Only use pin_memory for CUDA (not MPS)
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    
    return train_loader, test_loader, len(vocab), vocab


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (texts, labels, lengths) in enumerate(pbar):
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts, lengths).squeeze()
        loss = criterion(outputs, labels.float())
        
        # Backward pass
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels, lengths in tqdm(test_loader, desc="Evaluating"):
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            
            outputs = model(texts, lengths).squeeze()
            loss = criterion(outputs, labels.float())
            
            test_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train LSTM on IMDB sentiment')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--data-dir', type=str, default='./data/imdb', help='Data directory')
    parser.add_argument('--save-path', type=str, default='./checkpoints/pytorch/lstm_imdb.pth',
                        help='Path to save model checkpoint')
    parser.add_argument('--vocab-path', type=str, default='./checkpoints/pytorch/imdb_vocab.pt',
                        help='Path to save vocabulary')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader, test_loader, vocab_size, vocab = get_data_loaders(
        args.batch_size, 
        args.data_dir,
        args.max_length
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Save vocabulary
    os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True)
    torch.save(vocab, args.vocab_path)
    print(f"Vocabulary saved to: {args.vocab_path}")
    
    # Model
    print("\nInitializing model...")
    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
        padding_idx=0
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            save_checkpoint(model, optimizer, epoch, test_loss, args.save_path)
            print(f"  ✓ Best model saved (acc: {best_acc:.2f}%)")
        
        print()
    
    print(f"Training complete! Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
