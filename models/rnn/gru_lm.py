"""
GRU Language Model for WikiText-2

Stacked GRU architecture for word-level language modeling on Wikipedia text.
Tests sequential dependencies and compilation overhead for multi-layer RNNs.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRULanguageModel(nn.Module):
    """
    Stacked GRU for word-level language modeling.
    
    Architecture:
        Embedding -> Stacked GRU (2-3 layers) -> Dropout -> Linear (vocab projection)
    
    The model predicts the next word in a sequence given previous words.
    Total parameters: ~5M (with default settings)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        hidden_dim: int = 200,
        num_layers: int = 2,
        dropout: float = 0.5,
        tie_weights: bool = True,
        padding_idx: int = 0,
    ):
        """
        Initialize GRU language model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of GRU
            num_layers: Number of GRU layers (default: 2)
            dropout: Dropout probability (applied between GRU layers and before output)
            tie_weights: Whether to tie input embedding and output projection weights
            padding_idx: Index for padding token in vocabulary
        """
        super(GRULanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout layer (applied to GRU output before final projection)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Optionally tie embedding and output weights
        # This reduces parameters and often improves performance
        if tie_weights:
            if embedding_dim != hidden_dim:
                raise ValueError(
                    f"When tie_weights=True, embedding_dim must equal hidden_dim. "
                    f"Got embedding_dim={embedding_dim}, hidden_dim={hidden_dim}"
                )
            self.fc.weight = self.embedding.weight
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with uniform distribution."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        if not self.tie_weights:
            self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            hidden: Previous hidden state of shape (num_layers, batch_size, hidden_dim)
                   If None, initialized to zeros
            lengths: Actual lengths of sequences (before padding) of shape (batch_size,)
                    If provided, uses pack_padded_sequence for efficiency
            
        Returns:
            Tuple of:
                - Output logits of shape (batch_size, seq_len, vocab_size)
                - Final hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        batch_size, seq_len = x.size()
        
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Pack padded sequences if lengths are provided
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed_embedded, hidden)
            # Unpack sequences
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
        else:
            # Process sequences normally
            output, hidden = self.gru(embedded, hidden)
        
        # Apply dropout to GRU output
        output = self.dropout(output)
        
        # Project to vocabulary size
        # output: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def detach_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Detach hidden state from computation graph.
        
        Useful for truncated backpropagation through time (TBPTT).
        
        Args:
            hidden: Hidden state
            
        Returns:
            Detached hidden state
        """
        return hidden.detach()


def create_gru_lm(
    vocab_size: int,
    embedding_dim: int = 200,
    hidden_dim: int = 200,
    num_layers: int = 2,
    dropout: float = 0.5,
    tie_weights: bool = True,
    padding_idx: int = 0,
    pretrained: bool = False
) -> GRULanguageModel:
    """
    Create a GRU language model.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of GRU
        num_layers: Number of GRU layers
        dropout: Dropout probability
        tie_weights: Whether to tie embedding and output weights
        padding_idx: Index for padding token
        pretrained: Whether to load pretrained weights (not implemented)
        
    Returns:
        GRULanguageModel model
    """
    model = GRULanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        tie_weights=tie_weights,
        padding_idx=padding_idx
    )
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for GRU language model")
    
    return model


def compute_perplexity(model: GRULanguageModel, loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity is a standard metric for language models.
    Lower is better (perplexity of 1 is perfect).
    
    Args:
        model: GRU language model (unused, for API consistency)
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()


if __name__ == "__main__":
    # Test the model
    print("Testing GRU Language Model")
    print("=" * 50)
    
    # Model parameters
    vocab_size = 10000
    batch_size = 8
    seq_len = 35  # Common sequence length for language modeling
    
    # Create model
    model = create_gru_lm(
        vocab_size=vocab_size,
        embedding_dim=200,
        hidden_dim=200,
        num_layers=2,
        tie_weights=True
    )
    
    print(f"Model: {model.__class__.__name__}")
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from models.utils import get_model_info
    
    info = get_model_info(model)
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    print(f"Weight tying: {'Enabled' if model.tie_weights else 'Disabled'}")
    print()
    
    # Test forward pass
    print("Testing forward pass:")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        # Initialize hidden state
        hidden = model.init_hidden(batch_size)
        print(f"Initial hidden shape: {hidden.shape}")
        
        # Forward pass
        logits, hidden_out = model(x, hidden)
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Final hidden shape: {hidden_out.shape}")
        print()
        
        # Test with variable lengths
        print("Testing with variable-length sequences:")
        lengths = torch.randint(20, seq_len + 1, (batch_size,))
        print(f"Sequence lengths: {lengths.tolist()}")
        
        hidden = model.init_hidden(batch_size)
        logits_var, hidden_var = model(x, hidden, lengths)
        print(f"Output logits shape (variable): {logits_var.shape}")
        print()
        
        # Test perplexity computation
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        perplexity = compute_perplexity(model, loss.item())
        print(f"Sample loss: {loss.item():.4f}")
        print(f"Sample perplexity: {perplexity:.2f}")
        print()
        
        # Test hidden state detachment
        hidden_detached = model.detach_hidden(hidden_out)
        print(f"Hidden state detached: {not hidden_detached.requires_grad}")
    
    print("\n✓ GRU Language Model test successful!")
