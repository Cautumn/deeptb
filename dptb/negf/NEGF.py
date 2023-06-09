from typing import List
import torch
from dptb.negf.pole_summation import pole_maker
from dptb.negf.RGF import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.utils import quad
from ase.io import read
from dptb.negf.poisson import density2Potential, getImg
from dptb.negf.SCF import _SCF
from dptb.utils.constants import *
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class NEGF(object):
    def __init__(self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.negf_options = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)