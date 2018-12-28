# -*- coding: utf-8 -*-
import itertools
import typing

import attr
import numpy as np
from fs.base import FS
from torch import nn, optim
import torch


@attr.s
class DomainPair:
    trf_to: nn.Module = attr.ib()
    in_dom: nn.Module = attr.ib()


class CycleGanTrainer:
    def __init__(self,
                 a_dom: DomainPair,
                 b_dom: DomainPair,
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.a_dom = a_dom
        self.b_dom = b_dom

        gen_params = itertools.chain(self.a_dom.trf_to.parameters(),
                                     self.b_dom.trf_to.parameters())
        is_params = itertools.chain(self.a_dom.in_dom.parameters(),
                                    self.b_dom.in_dom.parameters())
        self.generator_trainer = optim.Adam(gen_params)
        self.is_trainer = optim.Adam(is_params)

    def make_batch(self, a_data: torch.Tensor, b_data: torch.Tensor) -> \
            typing.Tuple[torch.Tensor, torch.Tensor]:
        a_size = a_data.shape[0]
        b_size = b_data.shape[0]
        a_batch_ix = np.random.choice(a_size, size=self.batch_size)
        b_batch_ix = np.random.choice(b_size, size=self.batch_size)

        a_domain_draw = a_data[a_batch_ix]
        b_domain_draw = b_data[b_batch_ix]
        return a_domain_draw, b_domain_draw

    def step_discrim(self,
                     a_data: torch.Tensor,
                     b_data: torch.Tensor) -> None:
        self.is_trainer.zero_grad()
        self.generator_trainer.zero_grad()

        a_domain_draw, b_domain_draw = self.make_batch(a_data, b_data)

        b_from_a = self.a_dom.trf_to(a_domain_draw)
        a_from_b = self.b_dom.trf_to(b_domain_draw)
        how_b_is_b_from_a = self.b_dom.in_dom(b_from_a)
        how_a_is_a_from_b = self.a_dom.in_dom(a_from_b)
        how_b_is_b = self.b_dom.in_dom(b_domain_draw)
        how_a_is_a = self.a_dom.in_dom(a_domain_draw)

        loss_discrim = (
            nn.MSELoss()(1, how_a_is_a) +
            nn.MSELoss()(1, how_b_is_b) +
            nn.MSELoss()(0, how_a_is_a_from_b) +
            nn.MSELoss()(0, how_b_is_b_from_a)
        )

        loss_discrim.backward()
        self.is_trainer.step()

    def step_gen(self,
                 a_data: torch.Tensor,
                 b_data: torch.Tensor) -> None:
        self.is_trainer.zero_grad()
        self.generator_trainer.zero_grad()

        a_domain_draw, b_domain_draw = self.make_batch(a_data, b_data)

        b_from_a = self.a_dom.trf_to(a_domain_draw)
        a_from_b = self.b_dom.trf_to(b_domain_draw)

        how_b_is_b_from_a = self.b_dom.in_dom(b_from_a)
        how_a_is_a_from_b = self.a_dom.in_dom(a_from_b)

        a_from_b_from_a = self.b_dom.trf_to(self.a_dom.trf_to(a_domain_draw))
        b_from_a_from_b = self.a_dom.trf_to(self.b_dom.trf_to(b_domain_draw))

        loss_discrim = (
            nn.MSELoss()(1, how_a_is_a_from_b) +
            nn.MSELoss()(1, how_b_is_b_from_a))
        loss_cyc = (
            nn.MSELoss()(a_domain_draw, a_from_b_from_a) +
            nn.MSELoss()(b_domain_draw, b_from_a_from_b))

        print(loss_discrim)
        print(loss_cyc)
        total_gen_loss = loss_discrim + loss_cyc
        total_gen_loss.backward()
        self.generator_trainer.step()

    def save_sample(self,
                    a_data: torch.Tensor,
                    b_data: torch.Tensor,
                    results_fs: FS) -> None:
        import matplotlib.pyplot as plt
        with torch.no_grad():
            a_domain_draw, b_domain_draw = self.make_batch(a_data, b_data)
            a_domain_draw = a_domain_draw[:4]
            b_domain_draw = b_domain_draw[:4]

            b_from_a = self.a_dom.trf_to(a_domain_draw).numpy()
            a_from_b = self.b_dom.trf_to(b_domain_draw).numpy()

            a_orig = a_domain_draw.numpy()
            b_orig = b_domain_draw.numpy()

        a_and_b = np.concatenate([b_from_a, a_orig], 1).reshape((4*56, 28))
        b_and_a = np.concatenate([a_from_b, b_orig], 1).reshape((4*56, 28))

        with results_fs.open('a_sample.png', 'wb') as f:
            plt.matshow(a_and_b)
            plt.savefig(f, format='png')
            plt.close('all')

        with results_fs.open('b_sample.png', 'wb') as f:
            plt.matshow(b_and_a)
            plt.savefig(f, format='png')
            plt.close('all')
