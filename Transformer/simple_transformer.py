# Simple Transformer encoder example in PyTorch
# pip install torch -- if you don't have it already

import math
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# ---- Simple Transformer Encoder Model ----
class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=2,
        max_len=128,
    ):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,  # [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head (use CLS-style: average pooling over sequence)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: [batch_size, seq_len] of token indices
        src_key_padding_mask: [batch_size, seq_len] with True for PAD tokens (optional)
        """
        # Embed tokens
        x = self.embedding(x)  # [batch, seq, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)  # [batch, seq, d_model]

        # Transformer encoder
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [batch, seq, d_model]

        # Simple pooling: mean over sequence dimension
        x = x.mean(dim=1)  # [batch, d_model]

        # Classification logits
        logits = self.fc(x)  # [batch, num_classes]
        return logits


# ---- Example usage with dummy data ----
if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 1000 # how many tokens are in one sample
    batch_size = 4 # how many samples the model processes together in one step
    seq_len = 10 # how many tokens are in one sample
    num_classes = 2

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=num_classes,
        max_len=128,
    )

    # Fake input: random token indices
    x = torch.randint(0, vocab_size, (batch_size, seq_len))  # [B, T]

    # Fake labels
    y = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    logits = model(x)  # [B, num_classes]
    print("Logits shape:", logits.shape)

    # Simple training step
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
