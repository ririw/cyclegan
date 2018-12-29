# -*- coding: utf-8 -*-

"""Console script for cyclegan."""
import datetime
import sys

import click
import fs
from fs.base import FS
import torch
from tqdm import tqdm

from cyclegan import training, generators, discriminators, datasets, monitoring


@click.command()
@click.option('--cuda/--no-cuda', help='Use CUDA', default=False)
@click.option('--debug/--no-debug', help='debug mode', default=False)
def main(cuda: bool, debug: bool) -> int:
    a_dom = training.DomainPair(
        generators.FashionMNISTMNISTTransform(),
        discriminators.MNISTDiscriminator()
    )
    b_dom = training.DomainPair(
        generators.MNISTMNISTTransform(),
        discriminators.MNISTDiscriminator()
    )
    monitoring.Writer.init('./results/{}'.format(datetime.datetime.now()))

    train_a = datasets.mnist(download=True).train_data
    train_b = datasets.fmnist(download=True).train_data
    a_data = train_a.type(torch.float32).contiguous() / 255
    b_data = train_b.type(torch.float32).contiguous() / 255

    trainer = training.CycleGanTrainer(a_dom, b_dom, use_cuda=cuda)

    n_iter = 4 if debug else 8192
    with fs.open_fs('file://./results', create=True) as res_fs:
        for i in tqdm(range(n_iter), smoothing=0.6):
            monitoring.Writer.step = i
            try:
                train_step(a_data, b_data, i, res_fs, trainer)
            except KeyboardInterrupt:
                break

        with res_fs.open('weights.pkl', 'wb') as f:
            torch.save(a_dom, f)
            torch.save(b_dom, f)
    return 0


def train_step(a_data: torch.Tensor,
               b_data: torch.Tensor,
               i: int,
               res_fs: FS,
               trainer: training.CycleGanTrainer) -> None:
    trainer.step_discrim(a_data, b_data)
    trainer.step_gen(a_data, b_data)
    if i % 8 == 0:
        with res_fs.makedirs('{:04d}'.format(i), recreate=True) as step_fs:
            trainer.save_sample(a_data, b_data, step_fs)


if __name__ == "__main__":
    sys.exit(main())  # pylint: disable=no-value-for-parameter
