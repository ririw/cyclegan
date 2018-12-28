import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


class MNISTMNISTTransform(nn.Module):  # type: ignore
    def __init__(self: 'MNISTMNISTTransform') -> None:
        super().__init__()
        self.upconv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.resnet_blocks = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )
        self.downconv_block = nn.Conv2d(32, 1, 3, padding=1)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.upconv_block(x)
        x = self.resnet_blocks.forward(x)
        x = self.downconv_block(x)
        return x.squeeze(1)
