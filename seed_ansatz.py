import numpy as np


def _dirichlet_weight(rng: np.random.Generator, n: int, alpha: float = 1.0) -> np.ndarray:
    """
    Return w ∈ R^n, w ≥ 0, sum(w) = 1.
    Samples w via independent Gamma(alpha,1) and normalizing.
    - alpha = 1.0  → uniform over the simplex (flat with no bias).
    - alpha < 1.0  → spikier (few large entries, many near zero).
    - alpha > 1.0  → smoother (entries more even).
    Edge cases:
      n = 0 → empty vector, n = 1 → [1.0].
    """

    #Sanity checks to avoid useless computation (R^0 and R vectors)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    #Invalid parameter
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be > 0.")
    
    
    #Gamma sampled vector of R^n
    g = rng.gamma(shape=alpha, scale=1.0, size=n)
    
    #Sum of the vector 
    s = g.sum()
    
    # numerically, s > 0 almost surely for alpha>0; still guard:
    if s == 0.0:
        # extremely unlikely; fallback to a one-hot at a random index
        w = np.zeros(n, dtype=np.float64)
        w[rng.integers(n)] = 1.0
        return w
    return g / s


# --------- Validators (keep constraints honest) ----------

def _check_L_row_substochastic(L: np.ndarray, atol: float = 1e-12):
    # For each (x,i): sum_j L[x,i,j] ≤ 1
    row_sums = L.sum(axis=2)  # (d, Dl)
    if np.any(row_sums > 1 + atol):
        raise AssertionError("random_L: found a row with sum > 1.")
    if np.any(L < -atol) or np.any(L > 1 + atol):
        raise AssertionError("random_L: entries must lie in [0,1].")

def _check_R_col_substochastic(R: np.ndarray, atol: float = 1e-12):
    # For each (x,j): sum_i R[x,i,j] ≤ 1
    col_sums = R.sum(axis=1)  # (d, Dr)
    if np.any(col_sums > 1 + atol):
        raise AssertionError("random_R: found a column with sum > 1.")
    if np.any(R < -atol) or np.any(R > 1 + atol):
        raise AssertionError("random_R: entries must lie in [0,1].")

def _check_box_0_1(A: np.ndarray, atol: float = 1e-12):
    if np.any(A < -atol) or np.any(A > 1 + atol):
        raise AssertionError("random_A1: entries must lie in [0,1].")