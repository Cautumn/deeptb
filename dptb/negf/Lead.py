import torch
from typing import List
from dptb.negf.surface_green import selfEnergy
import logging
from dptb.negf.utils import update_kmap, update_temp_file
import os
import numpy as np

log = logging.getLogger(__name__)

"""The data output of the intermidiate result should be like this:
{each kpoint
    "e_mesh":[],
    "emap":[]
    "se":[se(e0), se(e1),...], 
    "sgf":[...e...]
}
There will be a kmap outside like: {(0,0,0):1, (0,1,2):2}, to locate which file it is to reads.
"""

            
            

            # get output



class Lead(object):
    def __init__(self, tab, hamiltonian, structure, result_path) -> None:
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.tab = tab
        self.voltage = self.structure.lead_options["voltage"]
        self.result_path = result_path


    def self_energy(self, kpoint, ee, eta_lead: float=1e-5, method: str="Lopez-Sancho"):
        assert len(np.array(kpoint).reshape(-1)) == 3
        # according to given kpoint and e_mesh, calculating or loading the self energy and surface green function to self.
        ik = update_kmap(self.result_path, kpoint=kpoint)
        SEpath = os.path.join(self.result_path, self.tab+"_SE_k"+str(ik)+".pth")

        HL, HLL, HDL, SL, SLL, SDL = self.hamiltonian.get_hs_lead(kpoint, tab=self.tab, V=self.voltage)

        def fn(ee):
            se_list = []
            gf_list = []
            for e in ee:
                se, gf = selfEnergy(
                ee=e,
                hL=HL,
                hLL=HLL,
                sL=SL,
                sLL=SLL,
                hDL=HDL,
                sDL=SDL,
                voltage=self.voltage,
                etaLead=eta_lead, 
                method=method
            )
                se_list.append(se)
                gf_list.append(gf)

            return {"se":se_list, "gf":gf_list}

        self.segf = update_temp_file(update_fn=fn, file_path=SEpath, ee=ee, tags=["se", "gf"], info="Computing Electrode Self-Energy")

    def sigmaLR2Gamma(self, se):
        return -1j * (se - se.conj())
    
    @property
    def se(self):
        return torch.stack(self.segf["se"])
    
    @property
    def gf(self):
        return torch.stack(self.segf["gf"])
    
    @property
    def gamma(self):
        return self.sigmaLR2Gamma(self.se)