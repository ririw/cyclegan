# -*- coding: utf-8 -*-

"""Console script for cyclegan."""
import logging
import sys

import click
import fs
import torch
from tqdm import tqdm

from cyclegan import training, generators, discriminators, datasets


@click.command()
def main() -> int:
    a_dom = training.DomainPair(
        generators.MNISTMNISTTransform(),
        discriminators.MNISTDiscriminator()
    )
    b_dom = training.DomainPair(
        generators.MNISTMNISTTransform(),
        discriminators.MNISTDiscriminator()
    )

    train_x = datasets.mnist(download=True).train_data
    train_y = datasets.mnist(download=True).train_labels
    a_data = (train_x[train_y < 5].type(torch.float32).contiguous()) / 255
    b_data = (train_x[train_y >= 5]).type(torch.float32).contiguous() / 255

    trainer = training.CycleGanTrainer(a_dom, b_dom)

    with fs.open_fs('file://./results', create=True) as res_fs:
        for i in tqdm(range(1024)):
            trainer.step_discrim(a_data, b_data)
            trainer.step_gen(a_data, b_data)
            if i % 4 == 0:
                with res_fs.makedirs('{:04d}'.format(i), recreate=True) as step_fs:
                    trainer.save_sample(a_data, b_data, step_fs)

                with res_fs.open('fweights.pkl', 'wb') as f:
                    torch.save(a_dom, f)
                    torch.save(b_dom, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
