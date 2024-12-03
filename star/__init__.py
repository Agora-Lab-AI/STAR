from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from loguru import logger

class TokenMixingStructure(Enum):
    """Token mixing structure types for LIV operators."""
    DIAGONAL = auto()
    LOW_RANK = auto() 
    SCALED_TOEPLITZ = auto()
    SEQUENTIAL_SEMI_SEPARABLE = auto()

class ChannelMixingStructure(Enum):
    """Channel mixing structure types."""
    DIAGONAL = auto()
    DENSE = auto()
    GROUPED = auto()

@dataclass
class LIVConfig:
    """Configuration for Linear Input-Varying (LIV) operators."""
    featurizer_class: int
    token_mixing: TokenMixingStructure
    sparsity_mask: bool
    nonlinearity: Optional[str]
    channel_mixing: ChannelMixingStructure
    expansion_factor: int = 1
    repeat_factor: int = 1
    
    
class DiagonalOperator(nn.Module):
    """Diagonal operator implementation."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        """Forward pass implementing diagonal scaling.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            features: Featurized input
            
        Returns:
            Scaled output
        """
        return x * self.scale


class Featurizer(nn.Module):
    """Featurizer module for LIV operators."""
    
    def __init__(self, config: LIVConfig, input_dim: int):
        """Initialize featurizer.
        
        Args:
            config: Featurizer configuration
            input_dim: Input dimension
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Build featurizer layers based on config
        self.layers = self._build_layers()
        
        logger.debug(f"Initialized featurizer with config: {config}")

    def _build_layers(self) -> nn.ModuleList:
        """Construct featurizer layers based on configuration.
        
        Returns:
            ModuleList containing featurizer layers
        """
        layers = nn.ModuleList()
        expanded_dim = self.input_dim * self.config.expansion_factor
        
        if self.config.channel_mixing == ChannelMixingStructure.DENSE:
            layers.append(nn.Linear(self.input_dim, expanded_dim))
        
        if self.config.token_mixing == TokenMixingStructure.SCALED_TOEPLITZ:
            # Add depthwise convolutions for Toeplitz structure
            layers.append(nn.Conv1d(
                expanded_dim, 
                expanded_dim,
                kernel_size=3,
                padding=1,
                groups=expanded_dim
            ))
            
        return layers

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Featurized tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x.repeat(1, 1, self.config.repeat_factor)

class LIVOperator(nn.Module):
    """Linear Input-Varying operator implementation."""
    
    def __init__(self, config: LIVConfig, dim: int):
        """Initialize LIV operator.
        
        Args:
            config: Operator configuration
            dim: Model dimension
        """
        super().__init__()
        self.config = config
        self.dim = dim
        
        self.featurizer = Featurizer(config, dim)
        self.operator = self._build_operator()
        
        if config.nonlinearity:
            self.nonlinearity = getattr(nn.functional, config.nonlinearity.lower())
        else:
            self.nonlinearity = None
            
        logger.info(f"Initialized LIV operator with config: {config}")

    # Update LIVOperator._build_operator method
    def _build_operator(self) -> nn.Module:
        """Build the core operator based on mixing structures.
        
        Returns:
            Core operator module
        """
        if self.config.token_mixing == TokenMixingStructure.LOW_RANK:
            return LowRankOperator(self.dim)
        elif self.config.token_mixing == TokenMixingStructure.SEQUENTIAL_SEMI_SEPARABLE:
            return RecurrentOperator(self.dim)
        elif self.config.token_mixing == TokenMixingStructure.DIAGONAL:
            return DiagonalOperator(self.dim)
        raise NotImplementedError(f"Unsupported token mixing: {self.config.token_mixing}")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Processed tensor
        """
        # Get features from featurizer
        features = self.featurizer(x)
        
        # Apply operator
        out = self.operator(x, features)
        
        # Apply nonlinearity if configured
        if self.nonlinearity:
            out = self.nonlinearity(out)
            
        return out

class LowRankOperator(nn.Module):
    """Low-rank operator implementation."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.to_qkv = nn.Linear(dim, 3 * dim)
        
    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        """Forward pass implementing low-rank attention.
        
        Args:
            x: Input tensor
            features: Featurized input
            
        Returns:
            Attention output
        """
        q, k, v = self.to_qkv(features).chunk(3, dim=-1)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attn, v)

class RecurrentOperator(nn.Module):
    """Sequential semi-separable (recurrent) operator."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(dim, 2 * dim)
        
    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        """Forward pass implementing gated recurrence.
        
        Args:
            x: Input tensor
            features: Featurized input
            
        Returns:
            Recurrent output
        """
        # Compute gates
        i, f = self.gate(features).chunk(2, dim=-1)
        i, f = torch.sigmoid(i), torch.sigmoid(f)
        
        # Apply recurrence
        h = torch.zeros_like(x)
        outputs = []
        
        for t in range(x.size(1)):
            h = i[:, t] * x[:, t] + f[:, t] * h
            outputs.append(h)
            
        return torch.stack(outputs, dim=1)

class STARBackbone(nn.Module):
    """STAR architecture backbone."""
    
    def __init__(
        self, 
        dim: int,
        depth: int,
        backbone_genome: List[List[int]],
        liv_configs: Dict[int, LIVConfig]
    ):
        """Initialize STAR backbone.
        
        Args:
            dim: Model dimension
            depth: Number of layers
            backbone_genome: List of 5-integer genome sequences
            liv_configs: Mapping of LIV class IDs to configs
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Initialize layers
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        
        # Track shared featurizers and feature groups
        self.shared_featurizers: Dict[int, Featurizer] = {}
        self.shared_features: Dict[int, Dict[str, Tensor]] = {}
        
        # Build layers from genome
        self._build_from_genome(backbone_genome, liv_configs)
        
        logger.info(f"Initialized STAR backbone with dim={dim}, depth={depth}")

    def _build_from_genome(
        self,
        genome: List[List[int]], 
        configs: Dict[int, LIVConfig]
    ) -> None:
        """Build backbone architecture from genome sequence.
        
        Args:
            genome: Backbone genome
            configs: LIV configurations
        """
        for genes in genome:
            liv_class = genes[0]
            featurizer_sharing = genes[1]
            feature_sharing = genes[3]
            
            # Get or create featurizer
            if featurizer_sharing in self.shared_featurizers:
                featurizer = self.shared_featurizers[featurizer_sharing]
            else:
                featurizer = Featurizer(configs[liv_class], self.dim)
                self.shared_featurizers[featurizer_sharing] = featurizer
                
            # Create LIV operator
            operator = LIVOperator(configs[liv_class], self.dim)
            operator.featurizer = featurizer
            
            self.layers.append(operator)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through backbone.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Processed tensor
        """
        for layer in self.layers:
            # Pre-norm residual
            normed = self.norm(x)
            out = layer(normed)
            x = out * normed + x
            
        return x

# Example usage
if __name__ == "__main__":
    # Configure logging
    logger.add("star.log", rotation="1 day")
    
    # Example configuration
    dim = 512
    depth = 24
    
    # Example genome and configs
    genome = [
        [1, 1, 1, 1, 1],  # SA-1
        [9, 1, 1, 1, 1],  # GMemless
        [1, 2, 1, 2, 1],  # SA-1 with sharing
    ]
    
    configs = {
        1: LIVConfig(  # SA-1
            featurizer_class=1,
            token_mixing=TokenMixingStructure.LOW_RANK,
            sparsity_mask=False,
            nonlinearity="softmax",
            channel_mixing=ChannelMixingStructure.GROUPED
        ),
        9: LIVConfig(  # GMemless
            featurizer_class=9,
            token_mixing=TokenMixingStructure.DIAGONAL,
            sparsity_mask=False,
            nonlinearity="silu",
            channel_mixing=ChannelMixingStructure.DENSE
        )
    }
    
    # Create model
    model = STARBackbone(dim, depth, genome, configs)
    
    # Test forward pass
    x = torch.randn(2, 1024, dim)
    out = model(x)
    
    logger.info(f"Output shape: {out.shape}")