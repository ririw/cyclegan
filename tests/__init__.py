# -*- coding: utf-8 -*-

"""Unit test package for cyclegan."""
import ast
import os
import torch.cuda

_CUDA_DISABLED = not ast.literal_eval(os.environ.get('DISABLE_CUDA', 'False'))
USE_CUDA = _CUDA_DISABLED and torch.cuda.is_available()
