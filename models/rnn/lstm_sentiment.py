"""
LSTM Sentiment Classifier for IMDB Reviews

Bidirectional LSTM for binary sentiment classification (positive/negative).
Handles variable-length sequences and tests recurrent operator optimization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMSentimentClassifier(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.
    
    Architecture:
        Embedding -> Bidirectional LSTM -> Dropout -> Linear -> Sigmoid
    
    The model handles variable-length sequences and uses padding.
    Total parameters: ~2M (with default settings)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        """
        Initialize LSTM sentiment classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            padding_idx: Index for padding token in vocabulary
        """
        super(LSTMSentimentClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (binary classification)
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            lengths: Actual lengths of sequences (before padding) of shape (batch_size,)
                    If None, assumes all sequences are full length
            
        Returns:
            Output logits of shape (batch_size, 1) for binary classification
        """
        batch_size = x.size(0)
        
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Pack padded sequences if lengths are provided
        if lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                lengths_cpu, 
                batch_first=True, 
                enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            # Unpack sequences
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, 
                batch_first=True
            )
        else:
            # Without packing (all sequences same length or no padding)
            output, (hidden, cell) = self.lstm(embedded)
        
        # Use the final hidden state from all layers and directions
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Concatenate forward and backward final hidden states from last layer
        if self.bidirectional:
            # Get last layer's forward and backward hidden states
            hidden_fwd = hidden[-2, :, :]  # Forward direction of last layer
            hidden_bwd = hidden[-1, :, :]  # Backward direction of last layer
            hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        else:
            hidden_concat = hidden[-1, :, :]  # Last layer hidden state
        
        # Apply dropout
        hidden_concat = self.dropout(hidden_concat)
        
        # Final linear layer
        logits = self.fc(hidden_concat)
        
        return logits
    
    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get predictions (0 or 1).
        
        Args:
            x: Input tensor
            lengths: Sequence lengths
            
        Returns:
            Predictions of shape (batch_size,)
        """
        logits = self.forward(x, lengths)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long().squeeze()
        return predictions


def create_lstm_sentiment(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    bidirectional: bool = True,
    padding_idx: int = 0,
    pretrained: bool = False
) -> LSTMSentimentClassifier:
    """
    Create an LSTM sentiment classifier.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        padding_idx: Index for padding token
        pretrained: Whether to load pretrained weights (not implemented)
        
    Returns:
        LSTMSentimentClassifier model
    """
    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        padding_idx=padding_idx
    )
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for LSTM sentiment classifier")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Sentiment Classifier")
    print("=" * 50)
    
    # Model parameters
    vocab_size = 10000
    batch_size = 8
    seq_len = 128
    
    # Create model
    model = create_lstm_sentiment(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        bidirectional=True
    )
    
    print(f"Model: {model.__class__.__name__}")
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from models.utils import get_model_info
    
    info = get_model_info(model)
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    print()
    
    # Test forward pass with variable lengths
    print("Testing forward pass with variable-length sequences:")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(50, seq_len + 1, (batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Sequence lengths: {lengths.tolist()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
        predictions = model.predict(x, lengths)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.tolist()}")
    print()
    
    # Test without length masking
    print("Testing forward pass without length masking:")
    logits_no_mask = model(x)
    print(f"Output shape (no masking): {logits_no_mask.shape}")
    
    print("\n✓ LSTM Sentiment Classifier test successful!")
