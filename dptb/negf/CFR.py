import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal




def ozaki_residues(M_cut:int=1000):
    """
    It computes the poles and residues of the Ozaki formulism.

    Parameters
    ----------
    M_cut (int (optional)): The cutoff, i.e. 2 * M_cut is dimension of the Ozaki matrix.

    Returns
    -------
    poles: The positive half of poles, in ascending order.
    res: The residues of positive half of poles.
    """
    if not isinstance(M_cut, int):
        M_cut = int(M_cut)
    # diagonal part of Ozaki matrix

    N_curt = int(2 * M_cut)
    diag = np.zeros(N_curt)
    # off-diagonal part of Ozaki matrix
    off_diag = np.array([.5 / np.sqrt((2. * n - 1) * (2. * n + 1)) for n in range(1, N_curt)])
    # The reciprocal of poles (eigenvalues) are in numerically ascending order, we just need the positive half.
    evals, evecs = eigh_tridiagonal(d=diag, e=off_diag, select='i', select_range=(N_curt // 2, N_curt - 1))
    # return poles in ascending order
    poles = np.flip(1. / evals)
    # compute residues
    res = np.flip(np.abs(evecs[0, :]) ** 2 / (4. * evals ** 2))

    return torch.from_numpy(poles), torch.from_numpy(res)


if __name__ == "__main__":
    p, r = ozaki_residues(M_cut=1000)

    print(p, r)
