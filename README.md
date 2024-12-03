


# STAR: Structured Token-mixing Adaptive Residual Networks


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


STAR is a novel neural network architecture that implements Linear Input-Varying (LIV) operators with structured token mixing patterns. This architecture provides an efficient approach to sequence modeling by combining different mixing structures with adaptive residual connections.

## Key Features

- Flexible token mixing structures (Diagonal, Low-Rank, Scaled-Toeplitz, Sequential Semi-Separable)
- Configurable channel mixing patterns (Diagonal, Dense, Grouped)
- Feature sharing mechanisms for improved efficiency
- Adaptive residual connections with pre-norm architecture
- Genome-based architecture specification

## Installation

```bash
pip install star-backbone
```

## Quick Start

```python
from star import STARBackbone, LIVConfig, TokenMixingStructure, ChannelMixingStructure

# Configure model
dim = 512
depth = 24

# Define genome
genome = [
    [1, 1, 1, 1, 1],  # SA-1
    [9, 1, 1, 1, 1],  # GMemless
    [1, 2, 1, 2, 1],  # SA-1 with sharing
]

# Configure operators
configs = {
    1: LIVConfig(
        featurizer_class=1,
        token_mixing=TokenMixingStructure.LOW_RANK,
        sparsity_mask=False,
        nonlinearity="softmax",
        channel_mixing=ChannelMixingStructure.GROUPED
    ),
    9: LIVConfig(
        featurizer_class=9,
        token_mixing=TokenMixingStructure.DIAGONAL,
        sparsity_mask=False,
        nonlinearity="silu",
        channel_mixing=ChannelMixingStructure.DENSE
    )
}

# Create model
model = STARBackbone(dim, depth, genome, configs)
```

## Architecture Details

### LIV Operators

The core building blocks are Linear Input-Varying (LIV) operators that combine:
- Token mixing structures for sequence interaction
- Channel mixing patterns for feature transformation
- Nonlinear activations
- Optional sparsity masks

### Token Mixing Structures

- **DIAGONAL**: Element-wise scaling
- **LOW_RANK**: Attention-like mechanisms with Q/K/V projections
- **SCALED_TOEPLITZ**: Convolution-based local mixing
- **SEQUENTIAL_SEMI_SEPARABLE**: Recurrent processing with gating

### Channel Mixing Types

- **DIAGONAL**: Independent channel scaling
- **DENSE**: Full channel interaction
- **GROUPED**: Group-wise channel mixing

## Genome Specification

Each layer is specified by a 5-integer sequence:
1. LIV operator class ID
2. Featurizer sharing group
3. Reserved
4. Feature sharing group  
5. Reserved

## Configuration

The `LIVConfig` dataclass specifies:
- `featurizer_class`: Integer ID for featurizer type
- `token_mixing`: TokenMixingStructure enum value
- `channel_mixing`: ChannelMixingStructure enum value
- `sparsity_mask`: Boolean for optional sparsity
- `nonlinearity`: Optional activation function name
- `expansion_factor`: Channel expansion multiplier
- `repeat_factor`: Feature repeat factor

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Open Pull Request

## License

MIT License. See LICENSE file for details.

## Citation

If you use STAR in your research, please cite:

```bibtex
@article{star2024,
  title={STAR: Structured Token-mixing Adaptive Residual Networks},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```