import nose.tools
import torch
from torch import optim, nn

from cyclegan import datasets, generators
from tests import USE_CUDA


def test_mnist_mnist() -> None:
    trf = generators.MNISTMNISTTransform()
    x = datasets.mnist().train_data[:32].type(torch.float32) / 255
    y = datasets.mnist().train_data[32:64].type(torch.float32) / 255

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(32):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)


def test_fmnist_mnist() -> None:
    trf = generators.FashionMNISTMNISTTransform()
    x = datasets.fmnist().train_data[:32].type(torch.float32) / 255
    y = datasets.fmnist().train_data[32:64].type(torch.float32) / 255

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(32):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)


def test_mnist_svhn() -> None:
    trf = generators.MnistSvhnTransform()
    x = datasets.mnist().train_data[:32].type(torch.float32) / 255
    y = torch.from_numpy(
        datasets.svhn().data[:32]).type(torch.float32) / 255

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(32):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)


def test_svhn_mnist() -> None:
    trf = generators.SvhnMnistTransform()
    x = torch.from_numpy(
        datasets.svhn().data[:32]).type(torch.float32) / 255
    y = datasets.mnist().train_data[:32].type(torch.float32) / 255

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(32):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)
