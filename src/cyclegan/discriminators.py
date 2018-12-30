import torch
from torch import nn


class MNISTDiscriminator(nn.Module):  # type: ignore
    def __init__(self: 'MNISTDiscriminator') -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 16, 5),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7744, 512),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        n_x = x.shape[0]
        x = x.reshape(n_x, -1)
        return self.dense_layers(x)


class SVHNDiscriminator(nn.Module):  # type: ignore
    def __init__(self: 'SVHNDiscriminator') -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 16, 5),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(10816, 512),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        n_x = x.shape[0]
        x = x.reshape(n_x, -1)
        return self.dense_layers(x)
