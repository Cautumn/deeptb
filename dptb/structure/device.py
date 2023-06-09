import ase
import warnings
import logging
import ase.neighborlist
from ase import Atoms
import numpy  as np
import re
from itertools import accumulate
import ase.io
from dptb.utils.constants import anglrMId,atomic_num_dict
from dptb.utils.tools import get_uniq_symbol, env_smoth
from dptb.utils.index_mapping import Index_Mapings
from dptb.structure.abstract_stracture import AbstractStructure
from dptb.structure.structure import BaseStruct



class Device(BaseStruct):
    def __init__(self, atom, format, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode:str='none', time_symm=True, stru_options={}):
        super(Device, self).__init__(atom, format, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode, time_symm)
        self.stru_options = stru_options

    
    def updata_struct(self, atom, format, onsitemode:str='none'):
        self.init_desciption()
        self.onsitemode = onsitemode
        self._read_struct_(atom,format=format)
        self.atom_symbols = np.array(self.struct.get_chemical_symbols(), dtype=str)
        self.atom_numbers = np.array(self.struct.get_atomic_numbers(), dtype=int)
        self.atomtype = get_uniq_symbol(atomsymbols=self.atom_symbols)
        self._projection_()
        self.proj_atom_symbols = self.projected_struct.get_chemical_symbols()
        self.proj_atom_numbers = self.projected_struct.get_atomic_numbers()
        self.proj_atom_neles_per = np.array([self.proj_atom_neles[ii] for ii in self.proj_atom_symbols])
        self.proj_atom_to_atom_id = np.array(list(range(len(self.atom_symbols))))[self.projatoms]
        self.atom_to_proj_atom_id = np.array(list(accumulate([int(i) for i in self.projatoms]))) - 1
        self.proj_atomtype = get_uniq_symbol(atomsymbols=self.proj_atom_symbols)
        self.get_bond(cutoff=self.cutoff,time_symm=self.time_symm)
        self.if_env_ready = False
        self.if_onsitenv_ready = False

        self.IndMap.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_strain_index_map, self.onsite_strain_num, self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=self.atomtype)
    
    def init_desciption(self):
        # init description
        super().init_description()

        self.l_lead_norbs = None
        self.r_lead_norbs = None
        self.l_boundary = None
        self.r_boundary = None
        self.device_norbs = None
        self.offsets = None

    def check_leads(self):
        pass

    def sort(self):
        pass