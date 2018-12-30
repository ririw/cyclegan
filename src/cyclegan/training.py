# -*- coding: utf-8 -*-
import itertools
import typing

import attr
import numpy as np
import torch
from torch import nn, optim
import torchvision
from fs.base import FS

from cyclegan import monitoring


@attr.s
class Domain:
    trf_out_of_domain: nn.Module = attr.ib()
    in_dom: nn.Module = attr.ib()

    def visualize_domain(self, x: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


class MNISTDomain(Domain):
    def visualize_domain(self, x: torch.Tensor) -> np.ndarray:
        x = x.unsqueeze(1)
        return torchvision.utils.make_grid(x).numpy()


class SVHNDomain(Domain):
    def visualize_domain(self, x: torch.Tensor) -> np.ndarray:
        return torchvision.utils.make_grid(x).numpy()


class CycleGanTrainer:
    def __init__(self,
                 a_dom: Domain,
                 b_dom: Domain,
                 batch_size: int = 32,
                 use_cuda: int = False):
        super().__init__()
        self._use_cuda = use_cuda
        self.batch_size = batch_size
        if use_cuda:
            self.a_dom = attr.evolve(
                a_dom,
                trf_out_of_domain=a_dom.trf_out_of_domain.cuda(),
                in_dom=a_dom.in_dom.cuda())
            self.b_dom = attr.evolve(
                b_dom,
                trf_out_of_domain=b_dom.trf_out_of_domain.cuda(),
                in_dom=b_dom.in_dom.cuda())
        else:
            self.b_dom = b_dom
            self.a_dom = a_dom

        gen_params = itertools.chain(self.a_dom.trf_out_of_domain.parameters(),
                                     self.b_dom.trf_out_of_domain.parameters())
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

        b_from_a = self.a_dom.trf_out_of_domain(a_domain_draw)
        a_from_b = self.b_dom.trf_out_of_domain(b_domain_draw)
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

        b_from_a = self.a_dom.trf_out_of_domain(a_domain_draw)
        a_from_b = self.b_dom.trf_out_of_domain(b_domain_draw)

        how_b_is_b_from_a = self.b_dom.in_dom(b_from_a)
        how_a_is_a_from_b = self.a_dom.in_dom(a_from_b)

        a_from_b_from_a = self.b_dom.trf_out_of_domain(
            self.a_dom.trf_out_of_domain(a_domain_draw))
        b_from_a_from_b = self.a_dom.trf_out_of_domain(
            self.b_dom.trf_out_of_domain(b_domain_draw))

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
        with torch.no_grad():
            a_domain_draw, b_domain_draw = self.make_batch(a_data, b_data)
            a_domain_draw = a_domain_draw[:4]
            b_domain_draw = b_domain_draw[:4]

            b_from_a = self.a_dom.trf_out_of_domain(a_domain_draw)
            a_from_b = self.b_dom.trf_out_of_domain(b_domain_draw)

            a_from_b_from_a = self.b_dom.trf_out_of_domain(b_from_a)
            b_from_a_from_b = self.a_dom.trf_out_of_domain(a_from_b)

            a_orig = a_domain_draw
            b_orig = b_domain_draw

        if self._use_cuda:
            a_from_b_from_a = a_from_b_from_a.cpu()
            b_from_a = b_from_a.cpu()
            a_orig = a_orig.cpu()
            b_from_a_from_b = b_from_a_from_b.cpu()
            a_from_b = a_from_b.cpu()
            b_orig = b_orig.cpu()

        # Underscores ensure the right ordering on the
        # tensorboard dash.
        monitoring.Writer.add_image(
            'a/___original', self.a_dom.visualize_domain(a_orig))
        monitoring.Writer.add_image(
            'b/___original', self.b_dom.visualize_domain(b_orig))

        monitoring.Writer.add_image(
            'a/__b_from_a', self.b_dom.visualize_domain(b_from_a))
        monitoring.Writer.add_image(
            'b/__a_from_b', self.a_dom.visualize_domain(a_from_b))

        monitoring.Writer.add_image(
            'a/_a_from_b_from_a', self.a_dom.visualize_domain(a_from_b_from_a))
        monitoring.Writer.add_image(
            'b/_b_from_a_from_b', self.b_dom.visualize_domain(b_from_a_from_b))

        with results_fs.open('weights.pkl', 'wb') as f:
            torch.save(self.a_dom, f)
            torch.save(self.b_dom, f)
