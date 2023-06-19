from dptb.negf.RGF import recursive_gf
import logging
from dptb.utils.constants import eV
import torch
import os
from dptb.negf.utils import update_kmap, update_temp_file
from dptb.utils.constants import *

log = logging.getLogger(__name__)

class Device(object):
    def __init__(self, hamiltonian, structure, result_path, e_T=300) -> None:
        self.green = 0
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.result_path = result_path
        self.cdtype = torch.complex128
        self.device = "cpu"
        self.kBT = k * e_T / eV
        self.e_T = e_T
    
    def set_leadLR(self, lead_L, lead_R):
        self.lead_L = lead_L
        self.lead_R = lead_R

    def green_function(self, ee, kpoint, etaDevice=0., block_tridiagonal=True):
        assert len(np.array(kpoint).reshape(-1)) == 3
        self.ee = ee
        self.block_tridiagonal = block_tridiagonal
        self.kpoint = kpoint

        # if V is not None:
        #     HD_ = self.attachPotential(HD, SD, V)
        # else:
        #     HD_ = HD

        if os.path.exists(os.path.join(self.result_path, "POTENTIAL.pth")):
            self.V = torch.load(os.path.join(self.result_path, "POTENTIAL.pth"))
        else:
            self.V = None

        ik = update_kmap(self.result_path, kpoint=kpoint)
        GFpath = os.path.join(self.result_path, "GF_k"+str(ik)+".pth")
        
        hd, sd, hl, su, sl, hu = self.hamiltonian.get_hs_device(kpoint, self.V, block_tridiagonal)
        
        # for i, e in tqdm(enumerate(ee), desc="Compute green functions: "):
        
        tags = ["g_trans", \
               "grd", "grl", "gru", "gr_left", \
               "gnd", "gnl", "gnu", "gin_left", \
               "gpd", "gpl", "gpu", "gip_left"]
        
        def fn(ee):
            self.lead_L.self_energy(kpoint=kpoint, ee=ee)
            self.lead_R.self_energy(kpoint=kpoint, ee=ee)
            seL = self.lead_L.se
            seR = self.lead_R.se
            green = []
            green_ = {}
            if not block_tridiagonal:
                for k, e in enumerate(ee):
                    ans = recursive_gf(e, hl=[], hd=hd, hu=[],
                                        sd=sd, su=[], sl=[], 
                                        left_se=seL[k], right_se=seR[k], seP=None, s_in=None,
                                        s_out=None, eta=etaDevice)
                    green.append(ans)
            else:
                for k, e in enumerate(ee):
                    ans = recursive_gf(e, hl=hl, hd=hd, hu=hu,
                                        sd=sd, su=su, sl=sl, 
                                        left_se=seL[k], right_se=seR[k], seP=None, s_in=None,
                                        s_out=None, eta=etaDevice)
                    green.append(ans)
                    # green shape [[g_trans, grd, grl,...],[g_trans, ...]]
            
            for t in range(len(tags)):
                green_[tags[t]] = [green[x][t] for x in range(len(green))]

            return green_
        
        self.green = update_temp_file(update_fn=fn, file_path=GFpath, ee=ee, tags=tags, info="Computing Green's Function")

    def density(self):
        pass

    def current(self):
        v_L = self.lead_L.voltage
        v_R = self.lead_R.voltage


        pass

    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
    
    @property
    def dos(self):
        dos = 0
        sd = self.hamiltonian.get_hs_device(kpoint=self.kpoint, V=self.V, block_tridiagonal=self.block_tridiagonal)[1]
        for jj in range(len(self.grd[0])):
            temp = torch.stack([i[jj] for i in self.grd]) @ sd[jj]
            dos -= temp.imag.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) / pi

        return dos

    @property
    def tc(self):
        self.lead_L.self_energy(kpoint=self.kpoint, ee=self.ee)
        self.lead_R.self_energy(kpoint=self.kpoint, ee=self.ee)

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

        TC = torch.bmm(torch.bmm(gammaL, self.g_trans) ,torch.bmm(gammaR, self.g_trans.conj().permute(0,2,1))).real.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        return TC

    @property
    def g_trans(self):
        return torch.stack(self.green["g_trans"])
    
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
    