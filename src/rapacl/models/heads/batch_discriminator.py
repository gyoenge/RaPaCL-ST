import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class AdversarialDiscriminator(nn.Module):
    """Gradient reversal based discriminator for batch correction.

    Input:
        feature embedding [B, H]

    Output:
        batch/domain logits [B, n_cls]
    """
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        hidden_dim: int | None = None,
        nlayers: int = 2,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.LeakyReLU,
        reverse_grad: bool = True,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()

        if n_cls is None or n_cls < 2:
            raise ValueError(f"n_cls must be >= 2 for batch discriminator, got {n_cls}")

        self.d_model = d_model
        self.n_cls = n_cls
        self.hidden_dim = hidden_dim if hidden_dim is not None else d_model
        self.nlayers = nlayers
        self.reverse_grad = reverse_grad
        self.grl_lambda = grl_lambda

        layers = []
        in_dim = d_model

        for _ in range(max(nlayers - 1, 0)):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                activation(),
                nn.Dropout(dropout),
            ])
            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, n_cls))
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H]

        Returns:
            logits: [B, n_cls]
        """
        if self.reverse_grad:
            x = grad_reverse(x, self.grl_lambda)
        logits = self.discriminator(x)
        return logits