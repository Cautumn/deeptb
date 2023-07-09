import ase
import glob
import numpy as np
import os
from ase.io.trajectory import Trajectory
import torch
from dptb.structure.structure import BaseStruct
from dptb.dataprocess.processor import Processor
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import normalize_bandinfo
from ase import Atoms
import pickle

def read_data_mol(path, cutoff, proj_atom_anglr_m, proj_atom_neles, \
    onsitemode:str='uniform', time_symm=True, **kwargs):
    batch_size = kwargs['train']['batch_size']
    struct_list_sets, eigens_sets = [], []
    data_dirs = glob.glob(path + "/*")
    struct_list, eigens = [], []
    for ii in range(len(data_dirs)):
        with open(data_dirs[ii], 'rb') as f:
            loaded_dict = pickle.load(f)
        iatom = Atoms(loaded_dict["Name"], positions=loaded_dict["positions"])
        eigs = loaded_dict["eigvals"].reshape(1, -1)
        struct = BaseStruct(atom=iatom, format='ase', cutoff=cutoff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles, onsitemode=onsitemode, time_symm=time_symm)
        
        eigens.append(eigs)
        struct_list.append(struct)

        if ii % batch_size == batch_size - 1 or ii == len(data_dirs) - 1:
            eigens_sets.append(eigens)
            struct_list_sets.append(struct_list)
            eigens, struct_list = [], []
    return struct_list_sets, eigens_sets

def get_data_mol(path, batch_size, bond_cutoff, env_cutoff, onsite_cutoff, \
        proj_atom_anglr_m, proj_atom_neles, sorted_onsite="st", sorted_bond="st", sorted_env="st", \
        onsitemode:str='uniform', time_symm=True, device='cpu', dtype=torch.float32, if_shuffle=True, **kwargs):
    struct_list_sets, eigens_sets = read_data_mol(path, bond_cutoff, \
        proj_atom_anglr_m, proj_atom_neles, onsitemode, time_symm, **kwargs)
    assert len(struct_list_sets) == len(eigens_sets)
    processor_list = []
    for i in range(len(struct_list_sets)):
        processor_list.append(
            Processor(structure_list=struct_list_sets[i], batchsize=batch_size,
                        eigen_list=eigens_sets[i], device=device, 
                        dtype=dtype, env_cutoff=env_cutoff, onsite_cutoff=onsite_cutoff, onsitemode=onsitemode, 
                        sorted_onsite=sorted_onsite, sorted_bond=sorted_bond, sorted_env=sorted_env, 
                        if_shuffle = if_shuffle))
    return processor_list
   
def get_data(path, prefix, batch_size, bond_cutoff, env_cutoff, onsite_cutoff, proj_atom_anglr_m, proj_atom_neles, 
        sorted_onsite="st", sorted_bond="st", sorted_env="st", onsitemode:str='uniform', time_symm=True, device='cpu', dtype=torch.float32, if_shuffle=True, **kwargs):
    """
        input: data params
        output: processor
    """
    
    struct_list_sets, kpoints_sets, eigens_sets, bandinfo_sets, wannier_sets = read_data(path, prefix, bond_cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode, time_symm, **kwargs)
    assert len(struct_list_sets) == len(kpoints_sets) == len(eigens_sets) == len(bandinfo_sets) == len(wannier_sets)
    processor_list = []

    for i in range(len(struct_list_sets)):
        processor_list.append(
            Processor(structure_list=struct_list_sets[i], batchsize=batch_size,
                        kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], wannier_list=wannier_sets[i], device=device, 
                        dtype=dtype, env_cutoff=env_cutoff, onsite_cutoff=onsite_cutoff, onsitemode=onsitemode, 
                        sorted_onsite=sorted_onsite, sorted_bond=sorted_bond, sorted_env=sorted_env, if_shuffle = if_shuffle, bandinfo=bandinfo_sets[i]))
    
    return processor_list



