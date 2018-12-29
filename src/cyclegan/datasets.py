import fs.appfs
import torchvision

data_location = fs.appfs.UserCacheFS('cyclegan')

_mnist: torchvision.datasets.MNIST = None
_fmnist: torchvision.datasets.FashionMNIST = None
_svhn: torchvision.datasets.SVHN = None


def mnist(download: bool = False) -> torchvision.datasets.MNIST:
    global _mnist
    if _mnist is None:
        with data_location.makedirs('MNIST', recreate=True) as loc:
            _mnist = torchvision.datasets.MNIST(
                loc.getsyspath(''), download=download)
    return _mnist


def fmnist(download: bool = False) -> torchvision.datasets.FashionMNIST:
    global _fmnist
    if _fmnist is None:
        with data_location.makedirs('FashionMNIST', recreate=True) as loc:
            _fmnist = torchvision.datasets.FashionMNIST(
                loc.getsyspath(''), download=download)
    return _fmnist


def svhn(download: bool = False) -> torchvision.datasets.SVHN:
    global _svhn
    if _svhn is None:
        with data_location.makedirs('SVHN', recreate=True) as loc:
            _svhn = torchvision.datasets.SVHN(
                loc.getsyspath(''), download=download)
    return _svhn
