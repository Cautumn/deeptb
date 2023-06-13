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

class NEGF(object):
    def __init__(self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.results_path = run_opt.get('results_path')
        self.jdata = jdata
        self.results_path = run_opt.get('results_path')
        self.cdtype = torch.complex128
        self.device = "cpu"
        
        
        

        # get the parameters
        self.el_T = jdata["el_T"]
        self.e_fermi = jdata["e_fermi"]
        self.stru_options = j_must_have(jdata, "stru_options")
        self.pbc = self.stru_options["pbc"]
        if not any(self.pbc):
            self.kpoint = np.array([[0,0,0]])
        else:
            self.kpoint = kmesh_sampling(self.jdata["kmesh"])

        self.init_indices()
        self.unit = jdata["unit"]
        self.scf = jdata["scf"]
        self.properties = jdata["properties"]
        

        # computing the hamiltonian
        if os.path.exists(os.path.join(self.results_path, "HS.pth")) and jdata["read_HS"]:
            HS = torch.load(os.path.join(self.results_path, "HS.pth"))
        elif jdata["read_HS"]:
            log.error("The user should provide HS file")
            raise RuntimeError
        else:
            HS = {}

            HS["device"] = {}
            self.apiH.update_struct(self.structase, mode="device", stru_options=j_must_have(self.stru_options, "device"))
            self.atom_norbs = [self.apiH.structure.proj_atomtype_norbs[i] for i in self.apiH.structure.atom_symbols]
            self.apiH.get_HR()
            H, S = self.apiH.get_HK(kpoints=self.kpoint)
            d_start = int(np.sum(self.atom_norbs[:self.device_id[0]]))
            d_end = int(np.sum(self.atom_norbs)-np.sum(self.atom_norbs[self.device_id[1]:]))
            HD, SD = H[0,d_start:d_end, d_start:d_end], S[0, d_start:d_end, d_start:d_end]
            

            HS["device"].update({"HD":HD.cdouble()})
            HS["device"].update({"SD":SD.cdouble()})
            self.kBT = k * self.el_T / eV

            for kk in self.stru_options:
                if kk.startswith("lead"):
                    HS[kk] = {}
                    lead_id = [int(x) for x in self.stru_options.get(kk)["id"].split("-")]
                    l_start = int(np.sum(self.atom_norbs[:lead_id[0]]))
                    l_end = int(l_start + np.sum(self.atom_norbs[lead_id[0]:lead_id[1]]) / 2)
                    HL, SL = H[0,l_start:l_end, l_start:l_end], S[0, l_start:l_end, l_start:l_end] # lead hamiltonian
                    HDL, SDL = H[0,d_start:d_end, l_start:l_end], S[0,d_start:d_end, l_start:l_end] # device and lead's hopping
                    HS[kk].update({
                        "HL":HL.cdouble(), 
                        "SL":SL.cdouble(), 
                        "HDL":HDL.cdouble(), 
                        "SDL":SDL.cdouble()}
                        )

                    stru_lead = self.structase[lead_id[0]:lead_id[1]]
                    self.apiH.update_struct(stru_lead, mode="lead", stru_options=self.stru_options.get(kk))
                    self.apiH.get_HR()
                    h, s = self.apiH.get_HK(kpoints=self.kpoint)
                    nL = int(h.shape[1] / 2)
                    HLL, SLL = h[0, :nL, nL:], s[0, :nL, nL:] # H_{L_first2L_second}
                    assert (h[0, :nL, :nL] - HL).abs().max() < 1e-5 # check the lead hamiltonian get from device and lead calculation matches each other
                    HS[kk].update({
                        "HLL":HLL.cdouble(), 
                        "SLL":SLL.cdouble()}
                        )
            
            torch.save(obj=HS, f=os.path.join(self.results_path, "HS.pth"))
        
        self.HS = HS

        # computing parameters for NEGF
        self.e_mesh = torch.linspace(start=self.jdata["emin"]+self.e_fermi, end=self.jdata["emax"]+self.e_fermi, steps=int((self.jdata["emax"]-self.jdata["emin"])/self.jdata["espacing"]))
        if self.unit == "Hartree":
            self.e_mesh = self.e_mesh / (13.605662285137 * 2)
        elif self.unit == "eV":
            self.e_mesh = self.e_mesh
        elif self.unit == "Ry":
            self.e_mesh = self.e_mesh / 13.605662285137
        else:
            log.error("The unit name is not correct !")
            raise ValueError

    def init_indices(self):
        self.device_id = [int(x) for x in self.stru_options.get("device")["id"].split("-")]
        self.lead_ids = []
        for kk in self.stru_options:
            if kk.startswith("lead"):
                self.lead_ids.append([int(x) for x in self.stru_options.get(kk)["id"].split("-")])

    def compute(self):
        self.compute_electrode_self_energy()
        self.compute_green_function()
        self.compute_properties()
        


    def self_energy(
            self, 
            ee: List[torch.Tensor], 
            HL: torch.Tensor,
            HLL: torch.Tensor,
            SL: torch.Tensor,
            SLL: torch.Tensor,
            HDL = None,
            SDL = None,
            u: torch.Tensor = torch.scalar_tensor(0.), 
            etaLead : float = 1e-5,
            method: str = 'Lopez-Sancho'
            ):
        se_list = []

        for e in ee:
            se, _ = selfEnergy(hL=HL, hLL=HLL, sL=SL, sLL=SLL, hDL=HDL, sDL=SDL, ee=e, voltage=u,
                                etaLead=etaLead, method=method)
            se_list.append(se)

        se_list = torch.stack(se_list)

        return se_list
    
    def green_function(self, ee, HD, SD, SeE, V=None, etaDevice=0.):

        if V is not None:
            HD_ = self.attachPotential(HD, SD, V)
        else:
            HD_ = HD
        # for i, e in tqdm(enumerate(ee), desc="Compute green functions: "):
        green = []
        for i, e in enumerate(ee):
            ans = recursive_gf(e, hl=[], hd=[HD_], hu=[],
                                sd=[SD], su=[], sl=[], 
                                left_se=SeE["lead_L"][i], right_se=SeE["lead_R"][i], seP=None, s_in=None,
                                s_out=None, eta=etaDevice)
            green.append(ans)

        return green
    
    def compute_green_function(self):
        log.info(msg="Computing Green Functions")
        if self.jdata["scf"]:
            V = self.SCF()
        else:
            V = None

        if not self.jdata["read_GF"]:
            self.green = self.green_function(self.e_mesh, self.HS["device"]["HD"], self.HS["device"]["SD"], self.SeE, V=V, etaDevice=self.jdata["eta_device"])
            torch.save(obj=self.green, f=os.path.join(self.results_path, "./GF.pth"))
        else:
            self.green = torch.load(os.path.join(self.results_path, "./GF.pth"))

        return True
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
    
    def attachPotential(self, hd, sd, V):

        hd_V = []
        
        for i in range(len(hd)):
            
            hd_V.append(hd[i] - sd[i] * V[i])
            # hd_V.append(hd[i] - V[i])

        return hd_V
    
    def compute_electrode_self_energy(self):
        # compute SE for properties calculation:
        
        log.info(msg="Computing Electrode Self-Energy")

        if not self.jdata["read_SE"]:
            SeE = {}
            SeE["emesh"] = self.e_mesh
            for kk in self.HS:
                if kk.startswith("lead"):
                    SeE[kk] = self.self_energy(
                        self.e_mesh, 
                        HL=self.HS[kk]["HL"], 
                        HLL=self.HS[kk]["HLL"],
                        SL=self.HS[kk]["SL"],
                        SLL=self.HS[kk]["SLL"],
                        HDL=self.HS[kk]["HDL"],
                        SDL=self.HS[kk]["SDL"],
                        u=self.stru_options[kk]["voltage"],
                        etaLead=self.jdata["eta_lead"],
                        method=self.jdata["sgf_solver"]
                        )
        
            torch.save(obj=SeE, f=os.path.join(self.results_path, "./el_SE.pth"))
        else:
            SeE = torch.load(os.path.join(self.results_path, "./el_SE.pth"))
        self.SeE = SeE

        return True
    
    def compute_properties(self):
        
        out = {}
        for p in self.properties:
            log.info(msg="Computing {0}".format(p))
            out[p] = getattr(self, "compute_"+p)()

        torch.save(obj=out, f=os.path.join(self.results_path, "./properties.pth"))

    def compute_DOS(self):
        grd = [i[1] for i in self.green]
        return [self.DOS(grd=grd[i], SD=[self.HS["device"]["SD"]]) for i in range(len(grd))]
    
    def compute_TC(self):
        gtrains = [i[0] for i in self.green]
        return [self.TC(seL=self.SeE["lead_L"][i], seR=self.SeE["lead_R"][i], gtrains=gtrains[i]) for i in range(len(gtrains))]
    
    def DOS(self, grd, SD):
        dos = 0
        for jj in range(len(grd)):
            temp = grd[jj] @ SD[jj]
            dos -= torch.trace(temp.imag) / pi

        return dos

    def TC(self, seL, seR, gtrains):
        tx, ty = gtrains.shape
        lx, ly = seL.shape
        rx, ry = seR.shape
        x0 = min(lx, tx)
        x1 = min(rx, ty)

        gammaL = torch.zeros(size=(tx, tx), dtype=self.cdtype, device=self.device)
        gammaL[:x0, :x0] += self.sigmaLR2Gamma(seL)[:x0, :x0]
        gammaR = torch.zeros(size=(ty, ty), dtype=self.cdtype, device=self.device)
        gammaR[-x1:, -x1:] += self.sigmaLR2Gamma(seR)[-x1:, -x1:]

        TC = torch.trace(gammaL @ gtrains @ gammaR @ gtrains.conj().T).real

        return TC

    def sigmaLR2Gamma(self, se):
        return -1j * (se - se.conj())

    def SCF(self):
        pass