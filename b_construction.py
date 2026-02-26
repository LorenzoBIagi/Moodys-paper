import numpy as np
from utils_tenpy import wrap_site_tensor, tenpy_sites_and_svs
from tenpy.networks.mps import MPS


def site_tensor_from_Bi(Bi: dict, d: int, *, strict_keys: bool = True):
    """
    Costruisce T[x] = Bi[x] con fallback a zeri. T shape: (d, Dl, Dr).

    Se strict_keys=True, controlla che tutte le chiavi di Bi siano in [0, d-1].
    Richiede che Bi non sia vuoto.
    """
    if not Bi:
        raise ValueError("Bi non può essere vuoto: serve almeno una matrice per fissare (Dl, Dr).")

    if strict_keys:
        bad_keys = [k for k in Bi.keys() if not (0 <= k < d)]
        if bad_keys:
            raise ValueError(f"Chiavi fuori range [0, {d-1}]: {bad_keys}")

    # tutte le matrici devono avere la stessa shape 2D
    shapes = {np.shape(M) for M in Bi.values()}
    if len(shapes) != 1:
        raise ValueError(f"shape incoerenti in Bi: {shapes}")
    Dl, Dr = next(iter(shapes))
    if len((Dl, Dr)) != 2:
        raise ValueError(f"Le matrici in Bi devono essere 2D, shape trovata: {Dl, Dr}")

    # dtype comune (considera anche i fallback a 0)
    dtype = np.result_type(*[np.asarray(Bi.get(x, 0)).dtype for x in range(d)])

    T = np.zeros((d, Dl, Dr), dtype=dtype)
    zero_block = np.zeros((Dl, Dr), dtype=dtype)

    for x in range(d):
        Mx = np.asarray(Bi.get(x, zero_block), dtype=dtype)
        if Mx.shape != (Dl, Dr):
            raise ValueError(f"Bi[{x}] ha shape {Mx.shape}, atteso {(Dl, Dr)}")
        T[x] = Mx

    return T, Dl, Dr

def build_mps(B_list, d: int):
    # Controllo d intero positivo
    if d < 1 or int(d) != d:
        raise ValueError("`d` deve essere intero positivo.")
    d = int(d)

    N = len(B_list)
    if N == 0:
        raise ValueError("B_list non può essere vuoto.")

    tensors = []
    right_dims = []

    prev_Dr = None

    for i, Bi in enumerate(B_list):
        T, Dl, Dr = site_tensor_from_Bi(Bi, d)

        # Bordo sinistro
        if i == 0:
            if Dl != 1:
                raise ValueError(f"sito {i}: Dl deve essere 1 (bordo sinistro), trovato {Dl}")
        else:
            # Bond interno: Dl(i) deve coincidere con Dr(i-1)
            if Dl != prev_Dr:
                raise ValueError(
                    f"mismatch bond interno tra sito {i-1} e {i}: "
                    f"Dr[{i-1}]={prev_Dr}, Dl[{i}]={Dl}"
                )

        # Bordo destro
        if i == N - 1 and Dr != 1:
            raise ValueError(f"sito {i}: Dr deve essere 1 (bordo destro), trovato {Dr}")

        tensors.append(T)
        right_dims.append(Dr)
        prev_Dr = Dr

    A = [wrap_site_tensor(T) for T in tensors]
    sites, svs = tenpy_sites_and_svs(d, right_dims)

    return MPS(sites, A, svs, bc='finite', form='A')

def build_B_list(S0, K, N, d_op, m_op, u_op, pd, pu):
    pmid = 1 - pd - pu
    if not (0 <= pd <= 1 and 0 <= pu <= 1 and pd + pu <= 1):
        raise ValueError("Probabilità non valide: servono pd, pu >=0 e pd+pu <= 1")

    B_list = []

    # Sito 1: (1,2)
    B1 = {
        0: np.array([[ (S0/(N))*d_op*pd,  (S0/(N))*d_op*pd - K*pd ]]),
        1: np.array([[ (S0/(N))*m_op*pmid, ((S0/(N))*m_op - K)*pmid ]]),
        2: np.array([[ (S0/(N))*u_op*pu,  (S0/(N))*u_op*pu - K*pu ]]),
    }
    B_list.append(B1)

    # Siti 2..N-1: (2,2)  ← QUI MANCANO NEL TUO FILE
    Bi = {
            0: np.array([[ d_op*pd,  d_op*pd ],
                         [     0,                               pd ]]),
            1: np.array([[ m_op*pmid, m_op*pmid ],
                         [      0,                               pmid ]]),
            2: np.array([[ u_op*pu,  u_op*pu],
                         [    0,                                 pu ]]),
        }
    for i in range(2, N):
        B_list.append(Bi)

    # Sito N: (2,1)
    BN = {
        0: np.array([[ d_op*pd ],
                     [     pd ]],),
        1: np.array([[ m_op*pmid ],
                     [      pmid ]]),
        2: np.array([[ u_op*pu ],
                     [     pu ]]),
    }
    B_list.append(BN)
    return B_list
