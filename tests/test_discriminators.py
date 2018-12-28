import torch
import nose.tools
import numpy as np
from torch import optim, nn

from cyclegan import datasets, discriminators


def test_mnist_is_zero() -> None:
    trf = discriminators.MNISTDiscriminator()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(20):
        train_sample = np.random.choice(len(datasets.MNIST), size=128)

        x = datasets.MNIST.train_data[train_sample].type(torch.float32) / 255
        y = (datasets.MNIST.train_labels[train_sample] == 0).type(torch.float32)

        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.5)
