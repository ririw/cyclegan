import torch
from torch import nn


class MNISTDiscriminator(nn.Module):  # type: ignore
    def __init__(self: 'MNISTDiscriminator') -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(18432, 1),
            nn.Sigmoid(),
        )

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        n_x = x.shape[0]
        x = x.reshape(n_x, -1)
        return self.dense_layers(x)
