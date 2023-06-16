from dptb.negf.RGF import recursive_gf, Green
import logging
import torch
import os
from dptb.negf.utils import update_kmap, update_temp_file
from dptb.utils.constants import *

log = logging.getLogger(__name__)

class Device(object):
    def __init__(self, hamiltonian, structure, result_path) -> None:
        self.green = 0
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.result_path = result_path
    
    def set_leadLR(self, lead_L, lead_R):
        self.lead_L = lead_L
        self.lead_R = lead_R


    def green_function(self, ee, kpoint, etaDevice=0., block_tridiagonal=True):
        assert len(kpoint.reshape(-1)) == 3

        # if V is not None:
        #     HD_ = self.attachPotential(HD, SD, V)
        # else:
        #     HD_ = HD

        if os.path.exist(self.result_path, "POTENTIAL.pth"):
            V = torch.load(os.path.exist(self.result_path, "POTENTIAL.pth"))
        else:
            V = None

        ik = update_kmap(self.result_path, kpoint=kpoint)
        GFpath = os.path.join(self.result_path, "GF_k"+str(ik)+".pth")
        
        hd, sd, hl, su, sl, hu = self.hamiltonian.get_hs_device(kpoint, V, block_tridiagonal)
        seL = self.lead_L.se
        seR = self.lead_R.se
        # for i, e in tqdm(enumerate(ee), desc="Compute green functions: "):
        
        tags = ["g_trans", \
               "grd", "grl", "gru", "gr_left", \
               "gnd", "gnl", "gnu", "gin_left", \
               "gpd", "gpl", "gpu", "gip_left"]
        
        def fn(ee):
            green = []
            green_ = {}
            if not block_tridiagonal:
                for i, e in enumerate(ee):
                    ans = recursive_gf(e, hl=[], hd=[hd], hu=[],
                                        sd=[sd], su=[], sl=[], 
                                        left_se=seL[i], right_se=seR[i], seP=None, s_in=None,
                                        s_out=None, eta=etaDevice)
                    green.append(ans)
            else:
                for i, e in enumerate(ee):
                    ans = recursive_gf(e, hl=hl, hd=hd, hu=hu,
                                        sd=sd, su=su, sl=sl, 
                                        left_se=seL[i], right_se=seR[i], seP=None, s_in=None,
                                        s_out=None, eta=etaDevice)
                    green.append(ans)
            
            for i in range(len(tags)):
                green_[tags[i]] = [green[x][i] for x in range(len(green))]

            return green_
        
        self.green = update_temp_file(update_fn=fn, file_path=GFpath, ee=ee, tags=tags, info="Computing Green's Function")
        
    def density(self):
        pass

    def current(self):
        pass

    def dos(self, ee, kpoint):
        self.green_function(ee=ee, kpoint=kpoint)
        dos = 0
        sd = self.hamiltonian.get_hs_device()[1]
        for jj in range(len(self.grd)):
            temp = self.grd[jj] @ sd[jj]
            dos -= torch.trace(temp.imag) / pi

        return dos

    def tc(self, ee, kpoint):
        self.green_function(ee=ee, kpoint=kpoint)
        self.lead_L.self_energy(ee=ee, kpoint=kpoint)

        ne1, tx, ty = self.g_trans.shape
        ne2, lx, ly = self.lead_L.se.shape
        ne3, rx, ry = self.lead_R.se.shape
        assert ne1 == ne2 == ne3
        x0 = min(lx, tx)
        x1 = min(rx, ty)

        gammaL = torch.zeros(size=(ne1, tx, tx), dtype=self.cdtype, device=self.device)
        gammaL[:,:x0, :x0] += self.lead_L.gamma[:, :x0, :x0]
        gammaR = torch.zeros(size=(ne1, ty, ty), dtype=self.cdtype, device=self.device)
        gammaR[:, -x1:, -x1:] += self.lead_R.gamma[:, -x1:, -x1:]

        TC = torch.bmm(torch.bmm(gammaL, self.g_trans) ,torch.bmm(gammaR, self.g_trans.conj().T)).real.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        return TC

    @property
    def g_trans(self):
        return self.green["g_trans"]
    
    @property
    def grd(self):
        return self.green["grd"]
    
    @property
    def grl(self):
        return self.green["grl"]
    
    @property
    def gru(self):
        return self.green["gru"]
    
    @property
    def gr_left(self):
        return self.green["gr_left"]
    
    @property
    def gnd(self):
        return self.green["gnd"]
    
    @property
    def gnl(self):
        return self.green["gnl"]
    
    @property
    def gnu(self):
        return self.green["gnu"]
    
    @property
    def gnin_left(self):
        return self.green["gnin_left"]
    
    @property
    def gpd(self):
        return self.green["gpd"]
    
    @property
    def gpl(self):
        return self.green["gpl"]
    
    @property
    def gpu(self):
        return self.green["gpu"]
    
    @property
    def gip_left(self):
        return self.green["gip_left"]
    
    @property
    def norbs_per_atom(self):
        pass

    @property
    def positions(self):
        pass
    