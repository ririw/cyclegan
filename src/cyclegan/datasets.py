import fs.appfs
import torchvision

data_location = fs.appfs.UserCacheFS('cyclegan')

_mnist: torchvision.datasets.MNIST = None
_svhn = None


def mnist() -> torchvision.datasets.MNIST:
    global _mnist
    if _mnist is None:
        with data_location.makedirs('MNIST', recreate=True) as loc:
            _mnist = torchvision.datasets.MNIST(loc.getsyspath(''))
    return _mnist


def svhn() -> torchvision.datasets.SVHN:
    global _svhn
    if _svhn is None:
        with data_location.makedirs('SVHN', recreate=True) as loc:
            _svhn = torchvision.datasets.SVHN(loc.getsyspath(''))
    return _svhn
