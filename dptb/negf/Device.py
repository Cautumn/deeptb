from dptb.negf.RGF import recursive_gf
import logging
from dptb.utils.constants import eV
import torch
import os
from dptb.negf.utils import update_kmap, update_temp_file
from dptb.utils.constants import *
from scipy.integrate import simpson, trapezoid

log = logging.getLogger(__name__)

class Device(object):
    def __init__(self, hamiltonian, structure, result_path, e_T=300, efermi=0.) -> None:
        self.green = 0
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.result_path = result_path
        self.cdtype = torch.complex128
        self.device = "cpu"
        self.kBT = k * e_T / eV
        self.e_T = e_T
        self.efermi = efermi
        self.mu = self.efermi
    
    def set_leadLR(self, lead_L, lead_R):
        self.lead_L = lead_L
        self.lead_R = lead_R
        self.mu = 0.5*(self.lead_L.mu + self.lead_R.mu)

    def green_function(self, ee, kpoint, etaDevice=0., block_tridiagonal=True):
        assert len(np.array(kpoint).reshape(-1)) == 3
        if not isinstance(ee, torch.Tensor):
            ee = torch.tensor(ee)

        if hasattr(self, "__DOS__"):
            delattr(self, "__DOS__")
        if hasattr(self, "__DOS__"):
            delattr(self, "__TC__")


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
        s_in = [torch.zeros(i.shape).cdouble() for i in hd]
        
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
            gammaR = self.lead_R.gamma
            s01, s02 = s_in[0].shape
            se01, se02 = seL[0].shape
            idx0, idy0 = min(s01, se01), min(s02, se02)

            s11, s12 = s_in[-1].shape
            se11, se12 = seR[-1].shape
            idx1, idy1 = min(s11, se11), min(s12, se12)
            
            green = []
            green_ = {}
            for k, e in enumerate(ee):
                # s_in[0][:idx0,:idy0] = s_in[0][:idx0,:idy0] + seL[k][:idx0,:idy0]
                s_in[-1][-idx1:,-idy1:] = s_in[-1][-idx1:,-idy1:] + gammaR[k][-idx1:,-idy1:]
                ans = recursive_gf(e, hl=[], hd=hd, hu=[],
                                    sd=sd, su=[], sl=[], 
                                    left_se=seL[k], right_se=seR[k], seP=None, s_in=None,
                                    s_out=None, eta=etaDevice, chemiPot=self.mu)
                # s_in[0][:idx0,:idy0] = s_in[0][:idx0,:idy0] - seL[k][:idx0,:idy0]
                s_in[-1][-idx1:,-idy1:] = s_in[-1][-idx1:,-idy1:] - gammaR[k][-idx1:,-idy1:]
                green.append(ans)
                # green shape [[g_trans, grd, grl,...],[g_trans, ...]]
            
            for t in range(len(tags)):
                green_[tags[t]] = [green[x][t] for x in range(len(green))]

            return green_
        
        self.green = update_temp_file(update_fn=fn, file_path=GFpath, ee=ee, tags=tags, info="Computing Green's Function")

    def density(self):

        if self.lead_L.voltage != self.lead_R.voltage:
            assert self.gnd is not None

        pass

    def current(self) -> torch.Tensor:
        v_L = self.lead_L.voltage
        v_R = self.lead_R.voltage

        # check the energy grid satisfied the requirement
        emin = self.ee.min()
        emax = self.ee.max()
        vmin = min(v_L, v_R)
        vmax = max(v_L, v_R)
        if emin>vmin-4*self.kBT:
            log.error("The energy lower bound for current integration is not sufficient.")
            raise ValueError
        
        if emax<vmax+4*self.kBT:
            log.error("The energy upper bound for current integration is not sufficient.")
            raise ValueError

        I = simpson((self.fermi_dirac(self.ee - v_L) - self.fermi_dirac(self.ee - v_R)) * self.tc, self.ee)

        return I

    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
        
    
    def _cal_tc_(self):
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

        self.__TC__ = torch.bmm(torch.bmm(gammaL, self.g_trans) ,torch.bmm(gammaR, self.g_trans.conj().permute(0,2,1))).real.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    @property
    def dos(self):
        if not hasattr(self, "__DOS__"):
            self._cal_dos_()
            return self.__DOS__
           
        else:
            return self.__DOS__
            

    @property
    def tc(self):
        if not hasattr(self, "__TC__"):
            self._cal_tc_()

            return self.__TC__
        else:
            return self.__TC__
    
    def _cal_dos_(self):
        dos = 0
        sd = self.hamiltonian.get_hs_device(kpoint=self.kpoint, V=self.V, block_tridiagonal=self.block_tridiagonal)[1]
        for jj in range(len(self.grd[0])):
            temp = torch.stack([i[jj] for i in self.grd]) @ sd[jj]
            dos -= temp.imag.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) / pi

        self.__DOS__ = dos

    @property
    def g_trans(self):
        return torch.stack(self.green["g_trans"])
    
    @property
    def grd(self):
        print(self.green["grd"])
        return [torch.stack(i) for i in self.green["grd"]]
    
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
    