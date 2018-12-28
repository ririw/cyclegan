import nose.tools
import torch
import numpy as np

from cyclegan import datasets


def test_mnist_size() -> None:
    train_data = datasets.mnist().train_data
    v = train_data[0]
    nose.tools.assert_is_instance(v, torch.Tensor)
    nose.tools.assert_list_equal(
        list(v.size()),
        [28, 28])
    nose.tools.assert_equal(len(train_data), 60000)


def test_svhn_size() -> None:
    train_data = datasets.svhn().data
    v = train_data[0]
    nose.tools.assert_is_instance(v, np.ndarray)
    nose.tools.assert_list_equal(
        list(v.shape),
        [3, 32, 32])
    nose.tools.assert_equal(len(train_data), 73257)
