"""LayerRouter v1."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class MLP(nn.Module):
    """MLP.

    Example
    -------
    >>> module = MLP(embedding_dimension=256, condition_dimension=16)
    >>> x = torch.randn((1, 10, 256))
    >>> c = torch.randn((16,))
    >>> x = module(x, c)  # Shape: (1, 10, 256).
    """

    def __init__(
        self, 
        embedding_dimension: int, 
        condition_dimension: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension + condition_dimension,
                out_features=embedding_dimension * 3,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=embedding_dimension * 3,
                out_features=embedding_dimension,
            ),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, T, E).
        c : torch.Tensor
            The condition tensor (B, C).

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        c = c[None, None, :].repeat((x.size(0), x.size(1), 1))  # Make `c` catable.
        x = torch.cat((x, c), dim=-1)
        x = self.layers(x)

        return x


class LayerRouter(nn.Module):
    """LayerRouter.

    Example
    -------
    >>> module = LayerRouter(
    ...     embedding_dimension=256,
    ...     steps=16,
    ...     layers=(
    ...         ...
    ...     ),
    ... )
    >>> x = ...
    >>> x = module(x)
    """

    def __init__(
        self,
        embedding_dimension: int,
        steps: int,
        layers: Tuple[nn.Module],
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        steps : int
            The number of steps.
        layers : int
            The subsequent layers.
        """

        super().__init__()

        self.steps = steps
        self.layers = layers

        self.mlp_1 = MLP(
            embedding_dimension=embedding_dimension,
            condition_dimension=steps,
        )

        self.mlp_2 = MLP(
            embedding_dimension=embedding_dimension,
            condition_dimension=steps,
        )

        self.head = nn.Linear(
            in_features=embedding_dimension,
            out_features=len(layers),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, T, E).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor (B, T, E).
        """

        # TODO: refactoring required to avoid nested for-loop.
        # TODO: implement token-wise layer selection rather than sequence-wise.

        for step in torch.arange(self.steps):

            condition = F.one_hot(step, num_classes=self.steps).float()
            score = F.softmax(self.head(self.mlp_1(x, condition)), dim=-1)
            score = score.mean(dim=-2)
            index = (score + (score.argmax(dim=-1).view(-1, 1).detach() - score)).mean(dim=-1)  # STE.
            x = self.mlp_2(x, condition)

            # Reconstruct batch with x routed to chosen layers.

            for i in range(x.size(0)):
                index_i = int(index[i].item()) 
                x[i, ...] = self.layers[index_i](x[i, ...])

        return x
