import typing

import tensorboardX
import torch
from fs.base import FS
import numpy as np


class Writer:
    _writer: tensorboardX.SummaryWriter = None
    step = 0

    @classmethod
    def add_scalar(cls,
                   name: str,
                   val: float,
                   global_step: typing.Optional[int] = None,
                   walltime: typing.Optional[float] = None) -> None:
        if cls._writer is None:
            return None
        cls._writer.add_scalar(name, val, global_step, walltime)

    @classmethod
    def add_image(cls,
                  tag: str,
                  img_tensor: np.ndarray,
                  global_step: typing.Optional[int] = None,
                  walltime: typing.Optional[float] = None) -> None:
        if cls._writer is None:
            return None
        cls._writer.add_image(tag, img_tensor, global_step, walltime)

    @classmethod
    def add_graph(cls,
                  model: torch.nn.Module,
                  input_to_model: torch.Tensor = None,
                  verbose: bool = False) -> None:
        if cls._writer is None:
            return None
        cls._writer.add_graph(model, input_to_model, verbose)

    @classmethod
    def init(cls, out_dir: typing.Optional[str] = None) -> None:
        cls._writer = tensorboardX.SummaryWriter(out_dir)

    @classmethod
    def finish(cls, results_fs: typing.Optional[FS] = None) -> None:
        if cls._writer is not None:
            if results_fs is not None:
                with results_fs.open('tensors.json', 'w') as f:
                    cls._writer.export_scalars_to_json(f)
            cls._writer.close()
