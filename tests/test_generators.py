import torch
import nose.tools
from torch import optim, nn

from cyclegan import datasets, generators


def test_mnist_mnist() -> None:
    trf = generators.MNISTMNISTTransform()
    x = datasets.MNIST.train_data[:32].type(torch.float32) / 255
    y = datasets.MNIST.train_data[32:64].type(torch.float32) / 255

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(20):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.5)
