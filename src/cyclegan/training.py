# -*- coding: utf-8 -*-
import itertools
import typing

import attr
import numpy as np
from fs.base import FS
import torch
from torch import nn, optim

from cyclegan import monitoring


@attr.s
class DomainPair:
    trf_to: nn.Module = attr.ib()
    in_dom: nn.Module = attr.ib()


class CycleGanTrainer:
    def __init__(self,
                 a_dom: DomainPair,
                 b_dom: DomainPair,
                 batch_size: int = 32,
                 use_cuda: int = False):
        super().__init__()
        self._use_cuda = use_cuda
        self.batch_size = batch_size
        if use_cuda:
            self.a_dom = attr.evolve(
                a_dom,
                trf_to=a_dom.trf_to.cuda(),
                in_dom=a_dom.in_dom.cuda())
            self.b_dom = attr.evolve(
                b_dom,
                trf_to=b_dom.trf_to.cuda(),
                in_dom=b_dom.in_dom.cuda())
        else:
            self.b_dom = b_dom
            self.a_dom = a_dom

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

        if self._use_cuda:
            a_domain_draw = a_domain_draw.cuda()
            b_domain_draw = b_domain_draw.cuda()
        return a_domain_draw, b_domain_draw

    def step_discrim(self,
                     a_data: torch.Tensor,
                     b_data: torch.Tensor) -> None:
        # pylint: disable=too-many-locals
        self.is_trainer.zero_grad()
        self.generator_trainer.zero_grad()

        a_domain_draw, b_domain_draw = self.make_batch(a_data, b_data)

        b_from_a = self.a_dom.trf_to(a_domain_draw)
        a_from_b = self.b_dom.trf_to(b_domain_draw)
        how_b_is_b_from_a = self.b_dom.in_dom(b_from_a)
        how_a_is_a_from_b = self.a_dom.in_dom(a_from_b)
        how_b_is_b = self.b_dom.in_dom(b_domain_draw)
        how_a_is_a = self.a_dom.in_dom(a_domain_draw)

        mse = nn.MSELoss()
        loss_actual = mse(1, how_a_is_a) + mse(1, how_b_is_b)
        loss_gen = mse(0, how_a_is_a_from_b) + mse(0, how_b_is_b_from_a)

        loss_discrim = loss_actual + loss_gen
        loss_discrim.backward()

        monitoring.Writer.add_scalar(
            'discrim/actual', loss_actual.item(), monitoring.Writer.step)
        monitoring.Writer.add_scalar(
            'discrim/gen', loss_gen.item(), monitoring.Writer.step)
        monitoring.Writer.add_scalar(
            'discrim/total', loss_discrim.item(), monitoring.Writer.step)

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
            torch.mean(torch.abs(a_domain_draw - a_from_b_from_a)) +
            torch.mean(torch.abs(b_domain_draw - b_from_a_from_b)))
        total_gen_loss = loss_discrim + loss_cyc

        monitoring.Writer.add_scalar(
            'gen/discrim', loss_discrim.item(), monitoring.Writer.step)
        monitoring.Writer.add_scalar(
            'gen/cyc', loss_cyc.item(), monitoring.Writer.step)
        monitoring.Writer.add_scalar(
            'gen/total', total_gen_loss.item(), monitoring.Writer.step)

        total_gen_loss.backward()
        self.generator_trainer.step()

    def save_sample(self,
                    a_data: torch.Tensor,
                    b_data: torch.Tensor,
                    results_fs: FS) -> None:
        # pylint: disable=too-many-locals
        import matplotlib.pyplot as plt
        with torch.no_grad():
            a_domain_draw, b_domain_draw = self.make_batch(a_data, b_data)
            a_domain_draw = a_domain_draw[:4]
            b_domain_draw = b_domain_draw[:4]

            b_from_a = self.a_dom.trf_to(a_domain_draw)
            a_from_b = self.b_dom.trf_to(b_domain_draw)

            a_from_b_from_a = self.b_dom.trf_to(b_from_a)
            b_from_a_from_b = self.a_dom.trf_to(a_from_b)

            a_orig = a_domain_draw
            b_orig = b_domain_draw

        if self._use_cuda:
            a_from_b_from_a = a_from_b_from_a.cpu()
            b_from_a = b_from_a.cpu()
            a_orig = a_orig.cpu()
            b_from_a_from_b = b_from_a_from_b.cpu()
            a_from_b = a_from_b.cpu()
            b_orig = b_orig.cpu()

        a_orig = a_orig.numpy().reshape(4 * 28, 28)
        b_from_a = b_from_a.numpy().reshape(4 * 28, 28)
        a_from_b_from_a = a_from_b_from_a .numpy().reshape(4 * 28, 28)

        b_orig = b_orig.numpy().reshape(4 * 28, 28)
        a_from_b = a_from_b.numpy().reshape(4 * 28, 28)
        b_from_a_from_b = b_from_a_from_b.numpy().reshape(4 * 28, 28)

        a_and_b = np.concatenate([
            a_orig,
            b_from_a,
            a_from_b_from_a,
        ], 1)
        b_and_a = np.concatenate([
            b_orig,
            a_from_b,
            b_from_a_from_b,
        ], 1)

        a_and_b_rgb = np.concatenate([
            a_and_b[None, :, :],
            a_and_b[None, :, :],
            a_and_b[None, :, :],
            ], 0)
        b_and_a_rgb = np.concatenate([
            b_and_a[None, :, :],
            b_and_a[None, :, :],
            b_and_a[None, :, :],
        ], 0)

        step = monitoring.Writer.step
        monitoring.Writer.add_image('a_sample', a_and_b_rgb, step)
        monitoring.Writer.add_image('b_sample', b_and_a_rgb, step)

        with results_fs.open('a_sample.png', 'wb') as f:
            plt.matshow(a_and_b)
            plt.savefig(f, format='png')
            plt.close('all')

        with results_fs.open('b_sample.png', 'wb') as f:
            plt.matshow(b_and_a)
            plt.savefig(f, format='png')
            plt.close('all')

        with results_fs.open('weights.pkl', 'wb') as f:
            torch.save(self.a_dom, f)
            torch.save(self.b_dom, f)
