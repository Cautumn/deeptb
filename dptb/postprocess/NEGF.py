from typing import List
import torch
from dptb.negf.Areshkin import pole_maker
from dptb.negf.RGF import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.utils import quad, gauss_xw
from dptb.negf.CFR import ozaki_residues
from dptb.negf.hamiltonian import Hamiltonian
from dptb.negf.Areshkin import pole_maker
from dptb.negf.Device import Device
from dptb.negf.Lead import Lead
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
        self.kBT = k * self.el_T / eV
        self.e_fermi = jdata["e_fermi"]
        self.stru_options = j_must_have(jdata, "stru_options")
        self.pbc = self.stru_options["pbc"]
        if not any(self.pbc):
            self.kpoints = np.array([[0,0,0]])
        else:
            self.kpoints = kmesh_sampling(self.jdata["kmesh"])

        self.init_indices()
        self.unit = jdata["unit"]
        self.scf = jdata["scf"]
        self.block_tridiagonal = jdata["block_tridiagonal"]
        self.properties = jdata["properties"]
        

        # computing the hamiltonian
        self.hamiltonian = Hamiltonian(apiH=self.apiH, structase=self.structase, stru_options=jdata["stru_options"], result_path=self.results_path)
        with torch.no_grad():
            struct_device, struct_leads = self.hamiltonian.initialize(kpoints=self.kpoints)
        self.generate_energy_grid()

        device = Device(self.hamiltonian, struct_device, result_path=self.results_path, efermi=self.e_fermi)
        device.set_leadLR(
                lead_L=Lead(
                hamiltonian=self.hamiltonian, 
                tab="lead_L", 
                structure=struct_leads["lead_L"], 
                result_path=self.results_path,
                e_T=self.el_T,
                efermi=self.e_fermi, 
                voltage=self.jdata["stru_options"]["lead_L"]["voltage"]
            ),
            lead_R=Lead(
                hamiltonian=self.hamiltonian, 
                tab="lead_R", 
                structure=struct_leads["lead_R"], 
                result_path=self.results_path, 
                e_T=self.el_T,
                efermi=self.e_fermi, 
                voltage=self.jdata["stru_options"]["lead_R"]["voltage"]
            )
        )


    def generate_energy_grid(self):

        # computing parameters for NEGF
        
        cal_pole = False
        cal_neq_grid = False
        cal_int_grid = False
        cal_uni_grid = False

        if self.scf:
            v_list = [self.stru_options[i].get(["voltage"], None) for i in self.stru_options]
            v_list = [i for i in v_list if i is not None]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                cal_pole = True
                cal_neq_grid = True
        elif "density" in self.properties or "potential" in self.properties:
            cal_pole = True
            v_list = [self.stru_options[i].get(["voltage"], None) for i in self.stru_options]
            v_list = [i for i in v_list if i is not None]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                cal_neq_grid = True
        
        if "current" in self.properties:
            cal_int_grid = True

        if "DOS" in self.properties or "TC" in self.properties:
            cal_uni_grid = True
            self.uni_grid = torch.linspace(start=self.jdata["emin"], end=self.jdata["emax"], steps=int((self.jdata["emax"]-self.jdata["emin"])/self.jdata["espacing"]))

        if cal_pole:
            self.poles, self.residues = ozaki_residues(M_cut=self.jdata["M_cut"])
        
        if cal_neq_grid:
            xl = min(v_list)-4*self.kBT
            xu = max(v_list)+4*self.kBT
            self.neq_grid, self.neq_weight = gauss_xw(xl=xl, xu=xu, n=int((xu-xl)/self.jdata["espacing"]))

        if cal_int_grid:
            xl = min(v_list)-4*self.kBT
            xu = max(v_list)+4*self.kBT
            self.int_grid, self.int_weight = gauss_xw(xl=xl, xu=xu, n=int((xu-xl)/self.jdata["espacing"]))

        e_mesh = torch.concat([self.uni_grid, self.poles, self.neq_grid, self.int_grid], dim=0)
        e_mesh = torch.unique(e_mesh)
        self.e_mesh = e_mesh

    def compute(self):
        # compute the grid
        for k in self.kpoints:
            self.device.green_function(
                ee=self.e_mesh, 
                kpoint=k, 
                etaDevice=self.jdata["eta_device"], 
                block_tridiagonal=self.block_tridiagonal
                )
            self.compute_properties()
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
    
    def compute_properties(self):
        
        out = {}
        for p in self.properties:
            log.info(msg="Computing {0}".format(p))
            out[p] = getattr(self.device, p)()

        torch.save(obj=out, f=os.path.join(self.results_path, "./properties.pth"))

    def compute_DOS(self):
        dos = [self.DOS(grd=self.green[self.eindex[e]][1], SD=[self.HS["device"]["SD"]]) for e in self.uni_grid]
        return dos
    
    def compute_TC(self):
        tc = [self.TC(seL=self.SeE["lead_L"][self.eindex[e]], seR=self.SeE["lead_R"][self.eindex[e]], gtrains=self.green[self.eindex[e]][0]) for e in self.uni_grid]
        return tc
    
    def compute_current(self):
        self.device.green_function(ee=self.int_grid, kpoint=k, etaDevice=self.jdata["eta_device"], block_tridiagonal=self.block_tridiagonal)

    def SCF(self):
        pass