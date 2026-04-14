import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Simple projection head for contrastive learning.

    Supports both:
    - [B, H]
    - [B, V, H]

    and returns the same leading shape with projected last dim.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.use_mlp = use_mlp

        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, out_dim),
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [B, H] or [B, V, H]

        Returns:
            projected tensor with same leading dims:
                [B, P] or [B, V, P]
        """
        return self.proj(x)
