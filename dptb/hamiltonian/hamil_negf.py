import torch
import torch as th
import numpy as np
import logging
import re
from dptb.hamiltonian.transform_sk import RotationSK
from dptb.nnsktb.formula import SKFormula
from dptb.utils.constants import anglrMId
from dptb.hamiltonian.soc import creat_basis_lm, get_soc_matrix_cubic_basis

log = logging.getLogger(__name__)

class HamilNEGF(RotationSK):
    def __init__(self, dtype=torch.float32, device='cpu') -> None:
        super().__init__(rot_type=dtype, device=device)


    def cut(self, hamiltonian, indices=[]):
        pass