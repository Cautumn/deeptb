import torch
from dptb.negf.CFR import ozaki_residues
from dptb.negf.Areshkin import pole_maker
import numpy as np
from dptb.negf.utils import gauss_xw

class Density(object):
    def __init__(self) -> None:
        pass

    def integrate(self, device):
        pass

    def field(self):
        pass


class Ozaki(Density):
    def __init__(self, R, M_cut, n_gauss):
        super(Ozaki, self).__init__()
        self.poles, self.residues = ozaki_residues(M_cut)
        # here poles are in the unit of (e-mu) / kbT
        self.R = R
        self.n_gauss = n_gauss

    def integrate(self, device, kpoint):
        kBT = device.kBT
        # add 0th order moment
        poles = self.poles * kBT + device.lead_L.mu # left lead expression for rho_eq
        device.green_function([1j*self.R], kpoint=kpoint, block_tridiagonal=False)
        g0 = device.grd[0][0]
        device.green_function(poles, kpoint=kpoint, block_tridiagonal=False)
        grd = device.grd[0]
        DM_eq = 1.0j * self.R * g0
        # add higher order terms
        for i in range(grd.shape[0]):
            term = ((-4 * 1j * kBT) * grd[i] * self.residues[i]).imag
            DM_eq += term
        DM_eq = DM_eq.real

        if abs(device.lead_L.voltage - device.lead_R.voltage) < 1e-14:
            # calculating Non-equilibrium density
            xl, xu = min(device.lead_L.voltage, device.lead_R.voltage), max(device.lead_L.voltage, device.lead_R.voltage)
            xl, xu = xl - 4*kBT, xu + 4*kBT
            xs, wlg = gauss_xw(xl=torch.scalar_tensor(xl), xu=torch.scalar_tensor(xu), n=self.n_gauss)
            device.lead_L.self_energy(kpoint=kpoint, ee=xs)
            device.lead_R.self_energy(kpoint=kpoint, ee=xs)
            device.green_function(xs, kpoint=kpoint, block_tridiagonal=False)
            ggg = torch.bmm(torch.bmm(device.grd[0], device.lead_R.gamma), device.grd[0].conj().permute(0,2,1)) 
            ggg = ggg * (device.fermi_dirac(xs+device.mu-device.lead_R.mu) - device.fermi_dirac(xs+device.mu-device.lead_L.mu)).view(-1,1)
            DM_neq = (wlg.view(1,-1) * ggg).sum(dim=0)
        else:
            DM_neq = 0.

        return DM_eq, DM_neq