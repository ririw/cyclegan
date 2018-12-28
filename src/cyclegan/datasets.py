import fs.appfs
import torchvision

data_location = fs.appfs.UserCacheFS('cyclegan')
with data_location.makedirs('MNIST', recreate=True) as loc:
    MNIST = torchvision.datasets.MNIST(loc.getsyspath(''))

with data_location.makedirs('SVHN', recreate=True) as loc:
    SVHN = torchvision.datasets.SVHN(loc.getsyspath(''))
