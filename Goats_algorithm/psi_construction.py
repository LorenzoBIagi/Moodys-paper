from random_matrix import random_A, random_R, random_L
from utils_tenpy import wrap_site_tensor, tenpy_sites_and_svs
from tenpy.networks.mps import MPS


def build_mps_AR(d: int, N: int, D: int, seed=None):
    """
    Crea b(x) con bond dimensione D:
      A1: (d, 1, D)
      R2..R_{N-1}: (d, D, D)
      RN: (d, D, 1)
    """
    # controlli basilari
    if N < 2:
        raise ValueError("N deve essere >= 2 per avere A1 e almeno un R.")
    if d < 1 or int(d) != d:
        raise ValueError("`d` deve essere intero positivo.")
    if D < 1 or int(D) != D:
        raise ValueError("`D` deve essere intero positivo.")
    d = int(d)
    D = int(D)

    A1 = random_A(d, Dl=1, Dr=D, seed=seed)
    A1= A1.reshape(d,1,D)

    Rs = []
    for k in range(2, N):
        sk = None if seed is None else seed + k
        Rs.append(random_R(d, Dl=D, Dr=D, seed=sk))

    sN = None if seed is None else seed + N
    RN = random_R(d, Dl=D, Dr=1, seed=sN)

    tensors = [A1] + Rs + [RN]

    # wrap TenPy
    A = [wrap_site_tensor(T) for T in tensors]
    right_dims = [T.shape[2] for T in tensors]  # Dr per ciascun sito
    sites, svs = tenpy_sites_and_svs(d, right_dims)
    return MPS(sites, A, svs, bc='finite', form='A')

def build_mps_LAR(d: int, N: int, D: int, seed=None):
    """
    Costruisce una MPS del tipo

        L ... L  A  R ... R

    con:
      - se N dispari:   #L = #R
      - se N pari:      #L = #R + 1

    Shape dei tensori:
      - primo L:   (d, 1, D)
      - L interni: (d, D, D)
      - A centrale:
            (d, D, D) in generale
            (d, D, 1) se non ci sono R
      - R interni: (d, D, D)
      - ultimo R:  (d, D, 1)
    """

    # controlli basilari
    if N < 2:
        raise ValueError("N deve essere >= 2.")
    if d < 1 or int(d) != d:
        raise ValueError("`d` deve essere un intero positivo.")
    if D < 1 or int(D) != D:
        raise ValueError("`D` deve essere un intero positivo.")

    d = int(d)
    D = int(D)

    # numero di L e R
    nL = N // 2
    nR = (N - 1) // 2

    # utility per semi diversi ma riproducibili
    counter = 0
    def next_seed():
        nonlocal counter
        counter += 1
        return None if seed is None else seed + counter

    tensors = []

    # -------------------------
    # blocco sinistro di L
    # -------------------------
    if nL >= 1:
        # primo L di bordo
        tensors.append(random_L(d, Dl=1, Dr=D, seed=next_seed()))

        # eventuali L interni
        for _ in range(nL - 1):
            tensors.append(random_L(d, Dl=D, Dr=D, seed=next_seed()))

    # -------------------------
    # tensore centrale A
    # -------------------------
    Dl_A = D if nL > 0 else 1
    Dr_A = D if nR > 0 else 1
    tensors.append(random_A(d, Dl=Dl_A, Dr=Dr_A, seed=next_seed()))

    # -------------------------
    # blocco destro di R
    # -------------------------
    if nR >= 1:
        # R interni (tutti tranne l'ultimo)
        for _ in range(nR - 1):
            tensors.append(random_R(d, Dl=D, Dr=D, seed=next_seed()))

        # ultimo R di bordo
        tensors.append(random_R(d, Dl=D, Dr=1, seed=next_seed()))

    # wrap TenPy
    A = [wrap_site_tensor(T) for T in tensors]
    right_dims = [T.shape[2] for T in tensors]
    sites, svs = tenpy_sites_and_svs(d, right_dims)

    return MPS(sites, A, svs, bc='finite', form='A')


def build_mps_AR_bond(d: int, N: int, D: int, seed=None):
    """
    Crea un MPS psi con N siti (N dispari) e bond che crescono verso il centro
    come Dl -> min(Dl * d, D) e poi decrescono in modo simmetrico.

    Tensore sito i (0-based) ha shape (d, Dl_i, Dr_i).
    """
    # controlli basilari
    if N < 3 or N % 2 == 0:
        raise ValueError("N deve essere dispari e >= 3.")
    if d < 1 or int(d) != d:
        raise ValueError("`d` deve essere intero positivo.")
    if D < 1 or int(D) != D:
        raise ValueError("`D` deve essere intero positivo.")
    d = int(d)
    D = int(D)

    # half = indice del bond centrale
    half = N // 2  # per N=7 -> 3

    # costruiamo i bond da sinistra fino al centro
    bonds = [1]   # bond[0] = 1
    Dl = 1
    for k in range(1, half + 1):
        candidate = Dl * d
        Dr = candidate if candidate < D else D
        bonds.append(Dr)
        Dl = Dr

    # riflettiamo per avere simmetria: [1, ..., centro, ..., 1]
    # es: [1,3,9,10] -> [1,3,9,10,10,9,3,1]
    bonds = bonds + bonds[::-1]
    assert len(bonds) == N + 1  # N siti -> N+1 bond

    tensors = []

    # primo sito: Dl = bonds[0]=1, Dr = bonds[1]
    Dr0 = bonds[1]
    A1 = random_A(d=d, Dl = 1, Dr = Dr0, seed=seed)   # (d,1,Dr0)
    tensors.append(A1.reshape(d,1,Dr0))

    # siti 1..N-1
    for i in range(1, N):
        Dl_i = bonds[i]
        Dr_i = bonds[i + 1]
        sk = None if seed is None else seed + i
        tensors.append(random_R(d, Dl=Dl_i, Dr=Dr_i, seed=sk))  # (d, Dl_i, Dr_i)

    # wrap TenPy
    A = [wrap_site_tensor(T) for T in tensors]  # (d,Dl,Dr) -> (Dl,d,Dr)
    right_dims = [T.shape[2] for T in tensors]  # Dr per sito
    sites, svs = tenpy_sites_and_svs(d, right_dims)
    return MPS(sites, A, svs, bc='finite', form='A')
