from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger
import einops
import math


@dataclass
class LIVConfig:
    """Configuration for a Linear Input-Varying system

    Args:
        hidden_dim: Dimension of hidden states
        num_heads: Number of attention heads (if using attention)
        dropout: Dropout probability
        max_sequence_length: Maximum sequence length
        featurizer_type: Type of featurizer ('dense', 'toeplitz', 'diagonal')
        token_mixing_type: Type of token mixing ('attention', 'conv', 'recurrent')
        channel_mixing_type: Type of channel mixing ('dense', 'grouped', 'diagonal')
    """

    hidden_dim: int
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 2048
    featurizer_type: str = "dense"
    token_mixing_type: str = "attention"
    channel_mixing_type: str = "grouped"


class LIVFeaturizer(nn.Module):
    """Featurizer that maps input to operator parameters"""

    def __init__(self, config: LIVConfig):
        super().__init__()
        self.config = config

        # Input validation
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                f"Hidden dim {config.hidden_dim} must be divisible by num_heads {config.num_heads}"
            )

        self.head_dim = config.hidden_dim // config.num_heads

        # Initialize featurizer based on type
        if config.featurizer_type == "dense":
            self.q_proj = nn.Linear(
                config.hidden_dim, config.hidden_dim
            )
            self.k_proj = nn.Linear(
                config.hidden_dim, config.hidden_dim
            )
            self.v_proj = nn.Linear(
                config.hidden_dim, config.hidden_dim
            )
        elif config.featurizer_type == "toeplitz":
            kernel_size = min(7, config.max_sequence_length)
            self.conv = nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim * 3,
                kernel_size=kernel_size,
                padding="same",
                groups=config.num_heads,
            )
        elif config.featurizer_type == "diagonal":
            self.diag = nn.Parameter(
                torch.randn(3, config.hidden_dim)
            )
        else:
            raise ValueError(
                f"Unknown featurizer type: {config.featurizer_type}"
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Tuple of (query, key, value) tensors each of shape
            (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        logger.debug(f"Featurizer input shape: {x.shape}")

        if hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"Input hidden dim {hidden_dim} != config hidden dim {self.config.hidden_dim}"
            )

        if self.config.featurizer_type == "dense":
            # Linear projections
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        elif self.config.featurizer_type == "toeplitz":
            # Convolutional featurizer
            x_conv = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
            qkv = self.conv(x_conv)  # (batch, 3*hidden_dim, seq_len)
            qkv = qkv.transpose(
                1, 2
            )  # (batch, seq_len, 3*hidden_dim)
            q, k, v = qkv.chunk(3, dim=-1)

        else:  # diagonal
            q = x * self.diag[0]
            k = x * self.diag[1]
            v = x * self.diag[2]

        # Reshape to heads
        q = einops.rearrange(
            q, "b s (h d) -> b h s d", h=self.config.num_heads
        )
        k = einops.rearrange(
            k, "b s (h d) -> b h s d", h=self.config.num_heads
        )
        v = einops.rearrange(
            v, "b s (h d) -> b h s d", h=self.config.num_heads
        )

        # Apply dropout
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        logger.debug(
            f"Featurizer output shapes: q={q.shape}, k={k.shape}, v={v.shape}"
        )

        return q, k, v


class LIVOperator(nn.Module):
    """Linear Input-Varying operator that applies token and channel mixing"""

    def __init__(self, config: LIVConfig):
        super().__init__()
        self.config = config
        self.featurizer = LIVFeaturizer(config)

        if config.token_mixing_type == "attention":
            self.scale = 1.0 / math.sqrt(
                config.hidden_dim // config.num_heads
            )
        elif config.token_mixing_type == "conv":
            kernel_size = min(7, config.max_sequence_length)
            self.conv = nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=kernel_size,
                padding="same",
                groups=config.num_heads,
            )
        elif config.token_mixing_type == "recurrent":
            self.rnn = nn.GRU(
                config.hidden_dim // config.num_heads,
                config.hidden_dim // config.num_heads,
                batch_first=True,
            )
        else:
            raise ValueError(
                f"Unknown token mixing type: {config.token_mixing_type}"
            )

        # Output projection
        if config.channel_mixing_type == "dense":
            self.out_proj = nn.Linear(
                config.hidden_dim, config.hidden_dim
            )
        elif config.channel_mixing_type == "grouped":
            self.out_proj = nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=1,
                groups=config.num_heads,
            )
        elif config.channel_mixing_type == "diagonal":
            self.out_proj = nn.Parameter(
                torch.randn(config.hidden_dim)
            )
        else:
            raise ValueError(
                f"Unknown channel mixing type: {config.channel_mixing_type}"
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        logger.debug(f"LIV operator input shape: {x.shape}")

        # Get q,k,v from featurizer
        q, k, v = self.featurizer(x)

        if self.config.token_mixing_type == "attention":
            # Scaled dot-product attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                attn = attn.masked_fill(
                    ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)

        elif self.config.token_mixing_type == "conv":
            # Reshape for convolution
            out = einops.rearrange(v, "b h s d -> b (h d) s")
            out = self.conv(out)
            out = einops.rearrange(
                out, "b (h d) s -> b h s d", h=self.config.num_heads
            )

        else:  # recurrent
            # Run RNN over sequence for each head
            out = []
            for head in range(self.config.num_heads):
                head_out, _ = self.rnn(v[:, head])
                out.append(head_out)
            out = torch.stack(out, dim=1)

        # Combine heads and apply output projection
        out = einops.rearrange(out, "b h s d -> b s (h d)")

        if self.config.channel_mixing_type == "dense":
            out = self.out_proj(out)
        elif self.config.channel_mixing_type == "grouped":
            out = out.transpose(1, 2)
            out = self.out_proj(out)
            out = out.transpose(1, 2)
        else:  # diagonal
            out = out * self.out_proj

        out = self.dropout(out)

        logger.debug(f"LIV operator output shape: {out.shape}")

        return out


class STARModel(nn.Module):
    """Complete STAR model composed of multiple LIV operators"""

    def __init__(self, config: LIVConfig, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        # Stack of LIV operators
        self.layers = nn.ModuleList(
            [LIVOperator(config) for _ in range(num_layers)]
        )

        # Layer norm for each layer
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(config.hidden_dim)
                for _ in range(num_layers)
            ]
        )

        logger.info(
            f"Initialized STAR model with {num_layers} layers"
        )

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        logger.debug(f"STAR model input shape: {x.shape}")

        for i, (layer, norm) in enumerate(
            zip(self.layers, self.norms)
        ):
            # Pre-norm residual connection
            residual = x
            x = norm(x)
            x = layer(x, mask)
            x = x + residual

            logger.debug(f"Layer {i} output shape: {x.shape}")

        return x


# Example usage:
if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(lambda msg: print(msg), level="INFO")

    # Create model config
    config = LIVConfig(
        hidden_dim=256,
        num_heads=8,
        max_sequence_length=1024,
        featurizer_type="dense",
        token_mixing_type="attention",
        channel_mixing_type="grouped",
    )

    # Initialize model
    model = STARModel(config, num_layers=6)

    # Generate sample input
    batch_size = 2
    seq_len = 512
    x = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Forward pass
    output = model(x)
    logger.info(f"Final output shape: {output.shape}")
