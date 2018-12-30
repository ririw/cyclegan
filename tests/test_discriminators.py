import nose.tools
import numpy as np
import torch
from torch import optim, nn

from cyclegan import datasets, discriminators
from tests import USE_CUDA


def test_mnist_is_zero() -> None:
    trf = discriminators.MNISTDiscriminator()

    train_sample = np.random.choice(len(datasets.mnist()), size=32)
    raw_x = datasets.mnist().train_data[train_sample]
    raw_y = (datasets.mnist().train_labels[train_sample] == 0)
    x = raw_x.type(torch.float32) / 255
    y = raw_y.type(torch.float32)

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(128):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)


def test_fmnist_is_zero() -> None:
    trf = discriminators.MNISTDiscriminator()

    train_sample = np.random.choice(len(datasets.mnist()), size=32)
    raw_x = datasets.fmnist().train_data[train_sample]
    raw_y = (datasets.fmnist().train_labels[train_sample] == 0)
    x = raw_x.type(torch.float32) / 255
    y = raw_y.type(torch.float32)

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(128):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)


def test_svhn_is_zero() -> None:
    trf = discriminators.SVHNDiscriminator()

    train_sample = np.random.choice(len(datasets.mnist()), size=32)
    raw_x = torch.from_numpy(datasets.svhn().data[train_sample])
    raw_y = torch.from_numpy(
        (datasets.svhn().labels[train_sample] == 0).astype(int))
    x = raw_x.type(torch.float32) / 255
    y = raw_y.type(torch.float32)

    if USE_CUDA:
        trf = trf.cuda()
        x = x.cuda()
        y = y.cuda()

    trf_opt = optim.Adam(trf.parameters())
    errs = []
    for _ in range(128):
        trf_opt.zero_grad()
        y_gen = trf.forward(x)
        err = nn.MSELoss()(y_gen, y)
        err.backward()
        trf_opt.step()

        errs.append(err.item())

    nose.tools.assert_less(errs[-1], errs[0] * 0.75)
