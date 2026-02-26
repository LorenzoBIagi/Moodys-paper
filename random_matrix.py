import numpy as np
from seed_ansatz import _dirichlet_weight
from seed_ansatz import (
    _check_L_row_substochastic,
    _check_R_col_substochastic,
    _check_box_0_1,
)

def random_L(
    d: int, Dl: int, Dr: int, seed=None, *,
    dirichlet_alpha: float = 1.0, check: bool = True
) -> np.ndarray:
    """
    Substochastic L_k, shape (d, Dl, Dr).
    Constraint : for each physical index x and each ROW i,
        sum_j L[x, i, j] ≤ 1  (row-substochastic)
    Construction: for each (x,i), draw a total mass s ~ U[0,1) and a Dirichlet
    weight w over Dr entries; set the row L[x,i,:] = s * w.
    """
    rng = np.random.default_rng(seed)
    L = np.zeros((d, Dl, Dr), dtype=np.float64)
    
    #Avoid unnecessary computation if degenerate input
    if d == 0 or Dl == 0 or Dr == 0:
        return L

    #Populate the rows substocastically
    for x in range(d):
        for i in range(Dl):
            s = rng.random()  # total mass in [0,1)
            if s == 0.0:
                continue      # zero row
            w = _dirichlet_weight(rng, Dr, alpha=dirichlet_alpha)  # sums to 1
            L[x, i, :] = s * w                                    # row sum = s ≤ 1

    if check:
        _check_L_row_substochastic(L)
    return L

def random_R(
    d: int, Dl: int, Dr: int, seed=None, *,
    dirichlet_alpha: float = 1.0, check: bool = True
) -> np.ndarray:
    """
    Substochastic R_k, shape (d, Dl, Dr).
    Constraint : for each physical x and each COLUMN j,
        sum_i R[x, i, j] ≤ 1  (column-substochastic)
    Construction: for each (x,j), draw mass s ~ U[0,1) and a Dirichlet over Dl
    entries; set the column R[x, :, j] = s * w.
    """
    rng = np.random.default_rng(seed)
    R = np.zeros((d, Dl, Dr), dtype=np.float64)
    
    #Avoid unnecessary computation if degenerate input
    if d == 0 or Dl == 0 or Dr == 0:
        return R
    
    #Populate the columns substocastically
    for x in range(d):
        for j in range(Dr):
            s = rng.random()  # total mass in [0,1)
            if s == 0.0:
                continue      # zero column
            w = _dirichlet_weight(rng, Dl, alpha=dirichlet_alpha)  # sums to 1
            R[x, :, j] = s * w                                    # column sum = s ≤ 1

    if check:
        _check_R_col_substochastic(R)
    return R


# controllare generazione A
def random_A1(d: int, D: int, seed=None, *, check: bool = True) -> np.ndarray:
    """
    First site core A1, shape (d, 1, D).
    Continuous in [0,1] (uniform). To have occasional exact 1.0 values,
    we can optionally snap a tiny fraction to 1.0 after sampling.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 1.0, size=(d, 1, D)).astype(np.float64)  # in [0,1)
    
    # Optional: snap some values to 1.0 with tiny prob (disabled by default)
    # snap_mask = rng.random(A.shape) < 1e-4
    # A[snap_mask] = 1.0
    
    if check:
        _check_box_0_1(A)
    return A
