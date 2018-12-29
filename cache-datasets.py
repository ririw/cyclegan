import logging

import click
import fs.appfs
import torchvision


@click.command(help='Download datasets needed for project.')
def download_dataset():
    logging.basicConfig(level=logging.INFO)
    data_location = fs.appfs.UserCacheFS('cyclegan')

    with data_location.makedirs('MNIST', recreate=True) as mnist_loc:
        logging.info('Caching MNIST dataset to %s', mnist_loc.getsyspath(''))
        torchvision.datasets.MNIST(mnist_loc.getsyspath(''), download=True)

    with data_location.makedirs('FashionMNIST', recreate=True) as mnist_loc:
        logging.info('Caching FashionMNIST dataset to %s', mnist_loc.getsyspath(''))
        torchvision.datasets.FashionMNIST(mnist_loc.getsyspath(''), download=True)

    with data_location.makedirs('SVHN', recreate=True) as svhn_loc:
        logging.info('Caching SVHN dataset to %s', svhn_loc.getsyspath(''))
        torchvision.datasets.SVHN(svhn_loc.getsyspath(''), download=True)


if __name__ == '__main__':
    download_dataset()
