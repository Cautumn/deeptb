from typing import List
import torch
from dptb.negf.pole_summation import pole_maker
from dptb.negf.RGF import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.utils import quad, gauss_xw
from dptb.negf.CFR import ozaki_residues
from dptb.negf.pole_summation import pole_maker
from ase.io import read
from dptb.negf.poisson import density2Potential, getImg
from dptb.negf.SCF import _SCF
from dptb.utils.constants import *
from dptb.negf.utils import leggauss
import logging
import os
import torch.optim as optim
from dptb.utils.tools import j_must_have
from tqdm import tqdm
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling

'''
1. split the leads, the leads and contact, and contact. the atoms
'''
log = logging.getLogger(__name__)

class Hamiltonian(object):
    def __init__(self, apiH, structase, stru_options, result_path) -> None:
        self.apiH = apiH
        self.unit = apiH.unit
        self.structase = structase
        self.stru_options = stru_options
        self.result_path = result_path
        
        self.device_id = [int(x) for x in self.stru_options.get("device")["id"].split("-")]
        self.lead_ids = []
        for kk in self.stru_options:
            if kk.startswith("lead"):
                self.lead_ids.append([int(x) for x in self.stru_options.get(kk)["id"].split("-")])

        if self.unit == "Hartree":
            self.h_factor = 13.605662285137 * 2
        elif self.unit == "eV":
            self.h_factor = 1.
        elif self.unit == "Ry":
            self.h_factor = 13.605662285137
        else:
            log.error("The unit name is not correct !")
            raise ValueError

    def initialize(self, kpoints):

        HS_device = {}
        HS_leads = {}
        HS_device["kpoints"] = kpoints
        HS_leads["kpoints"] = kpoints

        self.apiH.update_struct(self.structase, mode="device", stru_options=j_must_have(self.stru_options, "device"))
        structure_device = self.apiH.structure
        self.atom_norbs = [self.apiH.structure[i] for i in self.apiH.structure.atom_symbols]
        self.apiH.get_HR()
        H, S = self.apiH.get_HK(kpoints=kpoints)
        d_start = int(np.sum(self.atom_norbs[:self.device_id[0]]))
        d_end = int(np.sum(self.atom_norbs)-np.sum(self.atom_norbs[self.device_id[1]:]))
        HD, SD = H[:,d_start:d_end, d_start:d_end], S[:, d_start:d_end, d_start:d_end]
        

        HS_device["device"].update({"HD":HD.cdouble()*self.h_factor})
        HS_device["device"].update({"SD":SD.cdouble()})
        
        structure_leads = {}
        for kk in self.stru_options:
            if kk.startswith("lead"):
                HS_leads[kk] = {}
                lead_id = [int(x) for x in self.stru_options.get(kk)["id"].split("-")]
                l_start = int(np.sum(self.atom_norbs[:lead_id[0]]))
                l_end = int(l_start + np.sum(self.atom_norbs[lead_id[0]:lead_id[1]]) / 2)
                HL, SL = H[:,l_start:l_end, l_start:l_end], S[:, l_start:l_end, l_start:l_end] # lead hamiltonian
                HDL, SDL = H[:,d_start:d_end, l_start:l_end], S[:,d_start:d_end, l_start:l_end] # device and lead's hopping
                HS_leads[kk].update({
                    "HL":HL.cdouble()*self.h_factor, 
                    "SL":SL.cdouble(), 
                    "HDL":HDL.cdouble()*self.h_factor, 
                    "SDL":SDL.cdouble()}
                    )

                stru_lead = self.structase[lead_id[0]:lead_id[1]]
                self.apiH.update_struct(stru_lead, mode="lead", stru_options=self.stru_options.get(kk))
                structure_leads[kk] = self.apiH.structure
                self.apiH.get_HR()
                h, s = self.apiH.get_HK(kpoints=kpoints)
                nL = int(h.shape[1] / 2)
                HLL, SLL = h[:, :nL, nL:], s[:, :nL, nL:] # H_{L_first2L_second}
                assert (h[:, :nL, :nL] - HL).abs().max() < 1e-5 # check the lead hamiltonian get from device and lead calculation matches each other
                HS_leads[kk].update({
                    "HLL":HLL.cdouble()*self.h_factor, 
                    "SLL":SLL.cdouble()}
                    )
        
        return HS_device, HS_leads, structure_device, structure_leads
    
    def get_hs_device(self, kpoint, V, block_tridiagonal=False):
        

        if block_tridiagonal:
            return hd, sd, hl, su, sl, hu
        else:
            return HD, SD, None, None, None, None
    
    def get_hs_lead(self, kpoint, tab, V):

        return hL, hLL, hDL, sL, sLL, sDL
        pass

    def read(self, kpoints):
        if not isinstance(kpoints, np.ndarray) or not isinstance(kpoints, torch.Tensor):
            kpoints = torch.tensor(kpoints)
        if len(kpoints.shape) == 1:
            kpoints = kpoints.reshape(1, -1)

        file_kpts = torch.load(os.path.join(self.result_path, "HS.pth"))["kpoints"]
        HS = {}
        indices = []
        for k in kpoints:
            for i, j in enumerate(file_kpts):
                if (k - j).abs().mean() < 1e-14:
                    indices.append(i)


        HS["device"]["HD"] = HS["device"]["HD"][indices]

        pass

    def attach_potential():
        pass

    def write(self):
        pass

    def get_hs_block_tridiagonal(self, kpoints):
        pass
