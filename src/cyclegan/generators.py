import torch
from torch import nn


class MNISTMNISTTransform(nn.Module):  # type: ignore
    def __init__(self: 'MNISTMNISTTransform') -> None:
        super().__init__()
        self.upconv_block = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 32, 7),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.downconv_block = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 7),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.ConvTranspose2d(32, 16, 5),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.ConvTranspose2d(16, 1, 3),
            nn.Sigmoid()
        )
        # pylint: disable=arguments-differ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.upconv_block(x)
        x = self.downconv_block(x)

        return x.squeeze(1)
