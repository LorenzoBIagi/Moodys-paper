from tensorial_derivative import tensorial_derivative
from update_rule import newupdate, update_A
from random_matrix import random_A, random_R, random_L
import numpy as np
import tenpy.linalg.np_conserved as npc

def double_conflict(b, psi, A_loc, direction, seed=None):

    def next_seed(i):
        return None if seed is None else seed + i
    
    N = psi.L
    psi1 = psi.copy()
    psi2 = psi.copy()

    if direction == 'left':

        ## PRIMO STEP DELL ALGORITMO IN CUI SI CALCOLA L*L*A*R*R NON SPOSTO
        G = tensorial_derivative(psi=psi1, b=b, site=A_loc).to_ndarray()
        Dl, d_phys, Dr = G.shape
        A = update_A(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)

        psi1.set_B(A_loc-1, npc.Array.from_ndarray_trivial(A, labels=['vL', 'p', 'vR']))

        G = tensorial_derivative(psi=psi1, b=b, site=A_loc+1).to_ndarray()       

        if A_loc != N - 1:
            Dl, d_phys, Dr = G.shape
            R = newupdate(G.reshape(Dl, Dr * d_phys).T).T.reshape(Dl, d_phys, Dr)
            psi1.set_B(A_loc, npc.Array.from_ndarray_trivial(R, labels=['vL', 'p', 'vR']))
        else: 
            Dl, d_phys = G.shape
            R = newupdate(G.T).T.reshape(Dl, d_phys, 1)
            psi1.set_B(A_loc, npc.Array.from_ndarray_trivial(R, labels=['vL', 'p', 'vR']))
            #A_loc = N-1, NON POSSO SPOSTARLO PER SODDISFARE CONSTRAINT -> ESCO
            return psi1, A_loc

        k1 = b.overlap(psi1)

        ## SECONDO STEP DELL ALGORITMO IN CUI SI CALCOLA L*L*L*L*A*R STO SPOSTANDO (No check boundary perchè prima)
        Dl, d_phys, Dr = psi.get_B(A_loc-1).to_ndarray().shape
        psi2.set_B(A_loc-1, npc.Array.from_ndarray_trivial(random_L(d=d_phys, Dl= Dl, Dr=Dr, seed=next_seed(1)), labels=['vL', 'p', 'vR']))
        Dl, d_phys, Dr = psi.get_B(A_loc).to_ndarray().shape
        psi2.set_B(A_loc, npc.Array.from_ndarray_trivial(random_A(d=d_phys, Dl= Dl, Dr=Dr, seed=next_seed(2)), labels=['vL', 'p', 'vR']))

        G = tensorial_derivative(psi=psi2, b=b, site=A_loc).to_ndarray()
        Dl, d_phys, Dr = G.shape
        L = newupdate(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
        psi2.set_B(A_loc-1, npc.Array.from_ndarray_trivial(L, labels=['vL', 'p', 'vR']))

        G = tensorial_derivative(psi=psi2, b=b, site=A_loc+1).to_ndarray()
        Dl, d_phys, Dr = G.shape
        A = update_A(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
        psi2.set_B(A_loc, npc.Array.from_ndarray_trivial(A, labels=['vL', 'p', 'vR']))


        k2 = b.overlap(psi2)     

        if k2 <= k1:
            return psi1, A_loc
        else:
            return psi2, A_loc + 1


    elif direction == 'right':

        ## NOI ARRIVIAMO CON QUESTA CONFIGURAZIONE LLLAR*R*R*

        # PRIMO STEP DELL ALGORITMO DI DESTRA IN CUI SI CALCOLA LLL*A*R*R* NON SPOSTO (OTTIMIZZO A e R)

        #OTTIMIZZO A
        G = tensorial_derivative(psi=psi1, b=b, site=A_loc).to_ndarray() 
        Dl, d_phys, Dr = G.shape
        A = update_A(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
        psi1.set_B(A_loc-1, npc.Array.from_ndarray_trivial(A, labels=['vL', 'p', 'vR']))
        
        #OTTIMIZZO L (CON BOUNDARY CHECK)
        G = tensorial_derivative(psi=psi1, b=b, site=A_loc-1).to_ndarray() 
        if A_loc != 2:
                Dl, d_phys, Dr = G.shape
                L = newupdate(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
                psi1.set_B(A_loc-2, npc.Array.from_ndarray_trivial(A, labels=['vL', 'p', 'vR']))
        else:
                d_phys, Dr = G.shape
                L =  newupdate(G).reshape(1, d_phys, Dr)
                psi1.set_B(A_loc-2, npc.Array.from_ndarray_trivial(A, labels=['vL', 'p', 'vR']))
                #SE A_loc è 2 NON POSSO PUSHARLO A SINISTRA, OTTIMIZZO E BASTA E LASCIO COSI'
                return psi1, A_loc
      
        k1 = b.overlap(psi1)

        # SECONDO STEP DELL ALGORITMO DI DESTRA IN CUI SI CALCOLA LLA*R*R*R*  SPOSTO 
        
        # CREO RANDOM R E RANDOM A 
        Dl, d_phys, Dr = psi.get_B(A_loc-1).to_ndarray().shape
        psi2.set_B(A_loc-1, npc.Array.from_ndarray_trivial(random_R(d=d_phys, Dl= Dl, Dr=Dr, seed=next_seed(1)), labels=['vL', 'p', 'vR']))
        Dl, d_phys, Dr = psi.get_B(A_loc-2).to_ndarray().shape
        psi2.set_B(A_loc-2, npc.Array.from_ndarray_trivial(random_A(d=d_phys, Dl= Dl, Dr=Dr, seed=next_seed(2)), labels=['vL', 'p', 'vR']))

        # OTTIMIZZO RANDOM R E RANDOM A
        
        G = tensorial_derivative(psi=psi2, b=b, site=A_loc).to_ndarray()
        Dl, d_phys, Dr = G.shape
        R = newupdate(G.reshape(Dl, Dr * d_phys).T).T.reshape(Dl, d_phys, Dr)
        psi2.set_B(A_loc-1, npc.Array.from_ndarray_trivial(R, labels=['vL', 'p', 'vR']))
        
        G = tensorial_derivative(psi=psi2, b=b, site=A_loc-1).to_ndarray()
        Dl, d_phys, Dr = G.shape
        A = update_A(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
        psi2.set_B(A_loc-2, npc.Array.from_ndarray_trivial(A, labels=['vL', 'p', 'vR']))

        k2 = b.overlap(psi2)     

        if k2 <= k1:
            return psi1, A_loc
        else:
            return psi2, A_loc - 1

    else:
        return 'Wrong Direction'
        