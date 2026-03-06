import numpy as np

def newupdate(
    G: np.ndarray,
    *,
    zero_if_nonpositive: bool = True,
    dtype=np.uint8,
) -> np.ndarray:
    """
    Row-wise version of newupdate.

    Given a 2D array G (shape m x n), return X (m x n) where each row i is
    one-hot at the index of the maximum entry of G[i, :]. If
    zero_if_nonpositive=True and the row's maximum <= 0, the whole row is set
    to zeros.

    - NaNs in G are treated as -inf (ignored for maxima).
    - Ties are broken by np.argmax's rule (first occurrence).
    """
    if G.ndim != 2:
        raise ValueError("G must be a 2D array (m x n).")

    # Treat NaNs as -inf so they never win the argmax
    G_clean = np.where(np.isnan(G), -np.inf, G)

    # Argmax per row (shape: (m,))
    idx = np.argmax(G_clean, axis=1)

    # Build one-hot result
    m, n = G_clean.shape
    X = np.zeros((m, n), dtype=dtype)
    if m > 0:
        X[np.arange(m), idx] = 1

    zero_rows = np.all(G == 0, axis=1)
    if np.any(zero_rows):
        X[zero_rows, :] = 0

    if zero_if_nonpositive:
        row_max = np.max(G_clean, axis=1)  # (m,)
        mask = row_max > 0                 # keep only rows with strictly positive max
        if not np.all(mask):
            # zero out rows that don't pass the positivity test
            X[~mask, :] = 0

    return X


def update_A(
    G: np.ndarray,
    *,
    dtype=np.uint8,
    nan_as_zero: bool = True,
) -> np.ndarray:
    """
    Entrywise update rule:
        X_ij = 1 if G_ij > 0
        X_ij = 0 otherwise

    If nan_as_zero=True, NaNs are treated as 0.
    """
    if nan_as_zero:
        G = np.where(np.isnan(G), 0.0, G)

    return (G > 0).astype(dtype)