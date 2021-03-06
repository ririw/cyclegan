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


class FashionMNISTMNISTTransform(nn.Module):  # type: ignore
    def __init__(self: 'MNISTMNISTTransform') -> None:
        super().__init__()
        self.upconv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.downconv_block = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.ConvTranspose2d(64, 32, 5),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.ConvTranspose2d(32, 1, 3),
            nn.Sigmoid()
        )

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.upconv_block(x)
        x = self.downconv_block(x)

        return x.squeeze(1)


class MnistSvhnTransform(nn.Module):  # type: ignore
    def __init__(self: 'MnistSvhnTransform') -> None:
        super().__init__()
        self.upconv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.downconv_block = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.ConvTranspose2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.ConvTranspose2d(64, 32, 5),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.ConvTranspose2d(32, 3, 3),
            nn.Sigmoid()
        )

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.upconv_block(x)
        x = self.downconv_block(x)

        return x


class SvhnMnistTransform(nn.Module):  # type: ignore
    def __init__(self: 'SvhnMnistTransform') -> None:
        super().__init__()
        self.upconv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.downconv_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
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
        x = self.upconv_block(x)
        x = self.downconv_block(x)

        x = x.squeeze(1)
        return x
