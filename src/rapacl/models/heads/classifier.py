import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Simple linear classifier for patch/radiomics features."""
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm

        self.norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        if num_classes <= 2:
            self.fc = nn.Linear(in_dim, 1)
        else:
            self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H]

        Returns:
            logits:
                [B, 1] for binary
                [B, C] for multiclass
        """
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
