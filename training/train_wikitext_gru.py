"""
Training script for GRU Language Model on WikiText-2 dataset.

"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.rnn.gru_lm import GRULanguageModel
from models.utils import count_parameters, save_checkpoint


class WikiTextDataset(Dataset):
    """WikiText-2 dataset wrapper for language modeling."""
    
    def __init__(self, texts, vocab, seq_length=35):
        self.vocab = vocab
        self.seq_length = seq_length
        
        # Tokenize all text
        print("Tokenizing text...")
        all_tokens = []
        for text in tqdm(texts):
            if text.strip():  # Skip empty lines
                tokens = text.lower().split()
                all_tokens.extend(tokens)
        
        # Convert to indices
        self.data = torch.tensor(
            [self.vocab.get(token, self.vocab['<UNK>']) for token in all_tokens],
            dtype=torch.long
        )
        print(f"Total tokens: {len(self.data)}")
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        # Input: tokens[idx:idx+seq_length]
        # Target: tokens[idx+1:idx+seq_length+1]
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


def build_vocab(texts, min_freq=3):
    """Build vocabulary from texts."""
    counter = Counter()
    for text in tqdm(texts, desc="Building vocabulary"):
        if text.strip():
            tokens = text.lower().split()
            counter.update(tokens)
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    
    # Add tokens above minimum frequency
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def get_data_loaders(batch_size: int = 32, data_dir: str = "./data/wikitext", seq_length: int = 35):
    """
    Create WikiText-2 data loaders.
    
    Args:
        batch_size: Batch size for training and testing
        data_dir: Directory containing WikiText data
        seq_length: Sequence length for language modeling
        
    Returns:
        train_loader, val_loader, test_loader, vocab_size, vocab
    """
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=data_dir)
    
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']
    test_texts = dataset['test']['text']
    
    # Build vocabulary from training set
    vocab = build_vocab(train_texts)
    
    # Create datasets
    train_dataset = WikiTextDataset(train_texts, vocab, seq_length)
    val_dataset = WikiTextDataset(val_texts, vocab, seq_length)
    test_dataset = WikiTextDataset(test_texts, vocab, seq_length)
    
    # Only use pin_memory for CUDA (not MPS)
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    return train_loader, val_loader, test_loader, len(vocab), vocab


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        
        # Reshape for loss calculation
        # outputs: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        total_tokens += targets.numel()
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        ppl = math.exp(min(avg_loss, 10))  # Cap to avoid overflow
        pbar.set_postfix({
            'loss': avg_loss,
            'ppl': ppl
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_ppl = math.exp(min(epoch_loss, 10))
    
    return epoch_loss, epoch_ppl


def evaluate(model, data_loader, criterion, device, desc="Evaluating"):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=desc):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(min(avg_loss, 10))
    
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description='Train GRU Language Model on WikiText-2')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=200, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--seq-length', type=int, default=35, help='Sequence length')
    parser.add_argument('--tie-weights', action='store_true', default=True, help='Tie embedding and output weights')
    parser.add_argument('--data-dir', type=str, default='./data/wikitext', help='Data directory')
    parser.add_argument('--save-path', type=str, default='./checkpoints/pytorch/gru_wikitext.pth',
                        help='Path to save model checkpoint')
    parser.add_argument('--vocab-path', type=str, default='./checkpoints/pytorch/wikitext_vocab.pt',
                        help='Path to save vocabulary')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_size, vocab = get_data_loaders(
        args.batch_size, 
        args.data_dir,
        args.seq_length
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Save vocabulary
    os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True)
    torch.save(vocab, args.vocab_path)
    print(f"Vocabulary saved to: {args.vocab_path}")
    
    # Model
    print("\nInitializing model...")
    model = GRULanguageModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        padding_idx=0
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device, "Validating")
        
        # Step scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_loss, args.save_path)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        
        print()
    
    # Final test evaluation
    print("Evaluating on test set...")
    test_loss, test_ppl = evaluate(model, test_loader, criterion, device, "Testing")
    print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
