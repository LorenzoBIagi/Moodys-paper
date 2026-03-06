import numpy as np
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
import tenpy.linalg.np_conserved as npc

def wrap_site_tensor(T: np.ndarray):
    """(d, Dl, Dr) -> npc.Array labels ['vL','p','vR'] shape (Dl, d, Dr)"""
    Ai = np.transpose(T, (1, 0, 2))  # (Dl, d, Dr)
    return npc.Array.from_ndarray_trivial(Ai, labels=['vL', 'p', 'vR'])

def tenpy_sites_and_svs(d: int, right_dims):
    N = len(right_dims)
    S = (d - 1) / 2.0
    site = SpinSite(S=S, conserve=None)
    lattice = Chain(L=N, site=site)
    sites = lattice.mps_sites()
    svs = [np.ones(1, dtype=float)]
    svs += [np.ones(right_dims[i], dtype=float) for i in range(N - 1)]
    svs += [np.ones(1, dtype=float)]
    return sites, svs