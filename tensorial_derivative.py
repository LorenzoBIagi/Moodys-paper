import numpy as np
from tenpy.algorithms.network_contractor import ncon


def contract_left(tensor, i, N):
    A = [tensor.get_B(k) for k in range(i-1)]   # each: (Dl, d, Dr)
    #print(len(A))

    if i == 1:
        return None   # oppure semplicemente "return" per terminare senza valore

    if i > N:
        raise ValueError(f"Indice i={i} maggiore del numero di siti N={N}.")
    
    if i <= 0:
        raise ValueError(f"Indice i={i} minore o uguale a 0.")


    # ncon index bookkeeping:
    # open legs use negative integers; positive integers are contracted
    # A0: (-1, -2,  1)
    # A1: ( 1, -3,  2)
    # A2: ( 2, -4,  3)
    # ...
    # A_{i-2}: (i-2, -(i), -(i+1))  # the final right bond stays open

    con_tensors = []
    con_indices = []
    if len(A) == 1:
        con_indices.append([-1, -2, -3])
        con_tensors.append(A[0])
    else:
        for k, Ak in enumerate(A):
            if k == 0:                                  # first site
                con_tensors.append(Ak)
                con_indices.append([-1, -2, 1])
            elif k < len(A) - 1:                        # middle sites
                con_tensors.append(Ak)
                con_indices.append([k, -(k+2), k+1])
            else:                                       # last of the block
                con_tensors.append(Ak)
                con_indices.append([k, -(k+2), -(k+3)])
    #print(con_indices)

    # result has open legs: [-1 (left=1), -2..-(i) (physicals), -(i+1) (right bond Di-1)]
    T = ncon(con_tensors, con_indices)
    # T.shape == (1, d, d, ..., d, D_{i-1})   # (i-1) copies of d

    return T

def contract_right(tensor, i, N):
    A = [tensor.get_B(k) for k in range(i,N)]   # each: (Dl, d, Dr)
    #print(len(A))

    if i == N:
        return None   # oppure semplicemente "return" per terminare senza valore

    if i > N:
        raise ValueError(f"Indice i={i} maggiore del numero di siti N={N}.")
    
    if i <= 0:
        raise ValueError(f"Indice i={i} minore o uguale a 0.")

    # ncon index bookkeeping:
    # open legs use negative integers; positive integers are contracted
    # Ai: (-1, -2,  1)
    # Ai+1: ( 1, -3,  2)
    # Ai+2: ( 2, -4,  3)
    # ...
    # AN-1: (i-2, -(i), -(i+1))  # the final right bond stays open

    con_tensors = []
    con_indices = []
    if len(A) == 1:
        con_indices.append([-1, -2, -3])
        con_tensors.append(A[0])
    else:
        for k, Ak in enumerate(A):
            if k == 0:                                  # first site
                con_tensors.append(Ak)
                con_indices.append([-1, -2, 1])
            elif k < len(A) - 1:                        # middle sites
                con_tensors.append(Ak)
                con_indices.append([k, -(k+2), k+1])
            else:                                       # last of the block
                con_tensors.append(Ak)
                con_indices.append([k, -(k+2), -(k+3)])
    #print(con_indices)

    # result has open legs: [-1 (left=1), -2..-(i) (physicals), -(i+1) (right bond Di-1)]
    T = ncon(con_tensors, con_indices)
    # T.shape == (1, d, d, ..., d, D_{i-1})   # (i-1) copies of d
    return T

def tensorial_derivative(psi, b, site):

    N_psi = psi.L
    N_b  = b.L
    if N_psi != N_b:
        raise ValueError(f"psi e b hanno lunghezze diverse: {N_psi} vs {N_b}")
    N = N_psi

    if not (1 <= site <= N):
        raise ValueError(f"site={site} fuori range [1, {N}]")
    
    UL = contract_left(tensor=psi,i=site,N=N)
    UR = contract_right(tensor=psi,i=site,N=N)
    BL = contract_left(tensor=b,i=site,N=N)
    BR = contract_right(tensor=b,i=site,N=N)

    if UL is not None and len(UL.shape) != len(BL.shape):
        raise ValueError(f"Dimensione di UL diversa da BL")
    if UR is not None and len(UR.shape) != len(BR.shape):
        raise ValueError(f"Dimensione di UR diversa da BR")

    if UL is not None:
        left_tensors = [UL,BL]
        left_links = [
            [-1] + [k for k in range(1,site)] + [-2], 
        [-3] + [k for k in range(1,site)] + [-4] 
        ]
        #print('left contraction links',left_links)
        left_contraction = ncon(left_tensors,left_links) 
    ## Il risultato è un tensore di dimensione (1,D_i;1,2)

    if UR is not None:
        right_tensors = [UR,BR]
        right_links = [
            [-1] + [k for k in range(1,N-site+1)] + [-2], 
        [-3] + [k for k in range(1,N-site+1)] + [-4] 
        ]
        #print('right contraction links',right_links)
        right_contraction = ncon(right_tensors,right_links) 
    ## Il risultato è un tensore di dimensione (D_i+1,1;2,1)

    if site > 1 and site < N:
        final_tensors = [left_contraction, b.get_B(site-1),right_contraction]
        final_links = [
            [-1] + [-2] + [-3] + [1],
            [1] + [-4] + [2],
            [-5] + [-6] + [2] + [-7]
        ]
        final_contraction = ncon(final_tensors,final_links)
        #print(final_links)
        #print(final_contraction.shape)

    elif site == 1:
        final_tensors = [b.get_B(site-1),right_contraction]
        final_links = [
            [-1] + [-2] + [1],
            [-3] + [-4] + [1] + [-5]
        ]
        final_contraction = ncon(final_tensors,final_links)
        #print(final_links)
        #print(final_contraction.shape)

    elif site == N:
        final_tensors = [left_contraction, b.get_B(site-1)]
        final_links = [
            [-1] + [-2] + [-3] + [1],
            [1] + [-4] + [-5],
        ]
        final_contraction = ncon(final_tensors,final_links)
        #print(final_links)
        #print(final_contraction.shape)

    return np.squeeze(final_contraction)  