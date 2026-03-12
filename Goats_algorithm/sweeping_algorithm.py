import numpy as np
import tenpy.linalg.np_conserved as npc

from psi_construction import build_mps_AR, build_mps_LAR
from tensorial_derivative import tensorial_derivative
from update_rule import newupdate
from conflict import double_conflict


def SweepingAlgorithm(b, d, D, err):
    N = b.L
    psi = build_mps_AR(d=d, N=N, D=D)
    print("PSI")
    for k in range(psi.L):
        psik = psi.get_B(k).to_ndarray()
        print(f"Site {k}, shape = {psik.shape}:\n{psik.transpose(1,0,2)}\n")


    # overlap iniziale
    k_ref = b.overlap(psi)
    print("First ansatz:",k_ref)
    currency = []
    currency.append(k_ref)
    steps = []
    type = []
    steps.append(1)
    type.append('-->')

    # primo update sul sito 1 (come prima)
    site = 1
    A_tilde = tensorial_derivative(psi=psi, b=b, site=site)
    A_prime = A_tilde.to_ndarray()
    print("First tensorial derivative",A_prime)
    L_out = newupdate(A_prime)
    d_phys, Dr = L_out.shape
    L = L_out.reshape(1, d_phys, Dr)
    print("First max",L)
    L_tenpy = npc.Array.from_ndarray_trivial(L, labels=['vL', 'p', 'vR'])
    psi.set_B(site - 1, L_tenpy)

    nstep = 1
    updates_since_check = 1   # abbiamo già fatto un update
    K = 4         # o N, 2*N, ecc.
    max_steps = 100000
    #print("PSI")
    #for k in range(psi.L):
    #    psik = psi.get_B(k).to_ndarray()
    #    print(f"Site {k}, shape = {psik.shape}:\n{psik.transpose(1,0,2)}\n")

    while nstep < max_steps:

        # sweep left -> right
        for i in range(N - 1):
            site = i + 1
            if site == 1 and nstep == 1:
                continue
     
            #print("PSI PRE DERIVATA STEP",nstep)
            #for k in range(psi.L):
            #    psik = psi.get_B(k).to_ndarray()
            #    print(f"Site {k}, shape = {psik.shape}:\n{psik.transpose(1,0,2)}\n")
            A_prime = tensorial_derivative(psi=psi, b=b, site=site).to_ndarray()
            #print("-----------")
            print("step:",nstep)
            print("site:",site)
            

            if 1 < site < N:
                Dl, d_phys, Dr = A_prime.shape
                L = newupdate(A_prime.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
                print("tensorial derivative",A_prime.transpose(1,0,2))
                print("maximizer",L.transpose(1,0,2))
                type.append('-->')
            else:
                d_phys, Dr = A_prime.shape
                L =  newupdate(A_prime).reshape(1, d_phys, Dr)
                print("tensorial derivative",A_prime)
                print("maximizer",L)
                type.append('<--')

            
            psi.set_B(i, npc.Array.from_ndarray_trivial(L, labels=['vL', 'p', 'vR']))
            #print("PSI")
            #for k in range(psi.L):
            #    psik = psi.get_B(k).to_ndarray()
            #    print(f"Site {k}, shape = {psik.shape}:\n{psik.transpose(1,0,2)}\n")

            nstep += 1
            updates_since_check += 1
            k_curr = b.overlap(psi)
            currency.append(k_curr)
            steps.append(site)
            print("current price",k_curr)
            print("reference price",k_ref)

            if updates_since_check >= K:
                
                if np.abs(k_curr - k_ref) <= err * max(np.abs(k_ref), 1e-12):
                    return psi, k_curr, nstep, currency, steps, type
                k_ref = k_curr
                updates_since_check = 0
            #print("----------")

        # sweep right -> left
        #print("SWEEP RIGHT")
        for j in range(N - 1, 0, -1):
            site = j + 1
            A_prime = tensorial_derivative(psi=psi, b=b, site=site).to_ndarray()
            #print("-----------")
            print("step:",nstep)
            print("site:",site)

            

            if 1 < site < N:
                Dl, d_phys, Dr = A_prime.shape
                R = newupdate(A_prime.reshape(Dl, Dr * d_phys).T).T.reshape(Dl, d_phys, Dr)
                #R = newupdate(A_prime.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(2,0,1)
                print("tensorial derivative",A_prime.transpose(1,0,2))
                print("maximizer",R.transpose(1,0,2))
                type.append('<--')
            else:
                Dl, d_phys = A_prime.shape
                R = newupdate(A_prime.T).T.reshape(Dl, d_phys, 1)
                #R = newupdate(A_prime).reshape(1, d_phys, Dr).transpose(2,1,0)
                print("tensorial derivative SPIKE",A_prime)
                print("maximizer",R.transpose(1,0,2))
                print(R.shape)
                type.append('-->')

            psi.set_B(j, npc.Array.from_ndarray_trivial(R, labels=['vL', 'p', 'vR']))

            nstep += 1
            updates_since_check += 1
            k_curr = b.overlap(psi)
            currency.append(k_curr)
            steps.append(site)
            print("current price",k_curr)
            print("reference price",k_ref)
            

            if updates_since_check >= K:
                
                if np.abs(k_curr - k_ref) <= err * max(np.abs(k_ref), 1e-12):
                    return psi, k_curr, nstep, currency, steps, type
                k_ref = k_curr
                updates_since_check = 0
            #print("----------")

    raise RuntimeError("SweepingAlgorithm did not converge within max_steps")

def Sweep_double_conflict(b, d, D, err, K):

    N = b.L
    psi = build_mps_LAR(d=d, N=N, D=D)
    A_loc = N // 2 + 1
    
    #print("PSI")
    #for k in range(psi.L):
    #    psik = psi.get_B(k).to_ndarray()
    #    print(f"Site {k}, shape = {psik.shape}:\n{psik.transpose(1,0,2)}\n")


    # overlap iniziale
    k_ref = b.overlap(psi)
    #print("First ansatz:",k_ref)
    currency = []
    currency.append(k_ref)
    steps = []
    type = []
    steps.append(1)
    type.append('-->')

    # primo update sul sito 1 (come prima)
    site = 1
    A_tilde = tensorial_derivative(psi=psi, b=b, site=site)
    A_prime = A_tilde.to_ndarray()
    #print("First tensorial derivative",A_prime)
    L_out = newupdate(A_prime)
    d_phys, Dr = L_out.shape
    L = L_out.reshape(1, d_phys, Dr)
    #print("First max",L)
    L_tenpy = npc.Array.from_ndarray_trivial(L, labels=['vL', 'p', 'vR'])
    psi.set_B(site - 1, L_tenpy)

    nstep = 1
    updates_since_check = 1   # abbiamo già fatto un update
    #K = K questo ci dice che K è uguale al K in input (o N, 2*N, ecc.)
    max_steps = 100000
    #print("PSI")
    #for k in range(psi.L):
    #    psik = psi.get_B(k).to_ndarray()
    #    print(f"Site {k}, shape = {psik.shape}:\n{psik.transpose(1,0,2)}\n")

    # SONO 4 STEP DENTRO IL WHILE : AVANTI E DIETRO PER PRIMA E DOPO IL CENTRO
    #2 CONFLITTI PER OGNI WHILE : AVANTI E DIETRO
    while nstep < max_steps:

        #OTTIMIZZO FINO AL CENTRO QUINDI HO L*L*...L*A
        for i in range(1, A_loc - 1):

            sweep = 'left'
            
            site = i + 1 ## quindi site va da 2 a Aloc-1
            
            G = tensorial_derivative(psi=psi, b=b, site=site).to_ndarray()
            print("step:",nstep)
            print("site:",site)


            Dl, d_phys, Dr = G.shape
            L = newupdate(G.transpose(1,0,2).reshape(Dl * d_phys, Dr)).reshape(d_phys, Dl, Dr).transpose(1,0,2)
            #print("tensorial derivative",G.transpose(1,0,2))
            #print("maximizer",L.transpose(1,0,2))
            type.append('-->')
            
            #GLI DO I PERCHè LA CONVENZIONE è -1
            psi.set_B(i, npc.Array.from_ndarray_trivial(L, labels=['vL', 'p', 'vR']))

            nstep += 1
            updates_since_check += 1
            k_curr = b.overlap(psi)
            currency.append(k_curr)
            steps.append(site)
            print("current price",k_curr)
            print("reference price",k_ref)

            if updates_since_check >= K:
                
                if np.abs(k_curr - k_ref) <= err * max(np.abs(k_ref), 1e-12):
                    return psi, k_curr, nstep, currency, steps, type
                k_ref = k_curr
                updates_since_check = 0

        
        #PRIMO CONFLITTO (A LOC è IN CONVENZIONE NOSTRA)
        psi, A_loc = double_conflict(b=b, psi=psi, A_loc=A_loc, direction = sweep)

        #ORA OTTIMIZZIAMO DA DOPO A ALLA FINE
        ## L*L*L*ARRR -- > L*L*L*A*R*RR , L*L*L*L*A*RR in un caso la R è già ottimizzata ma la riottimizziamo

        for i in range(A_loc, N):

            sweep = 'left'
            
            site = i + 1 ## quindi site va da A_loc + 1 a N

            G = tensorial_derivative(psi=psi, b=b, site=site).to_ndarray()
            #print("step:",nstep)
            #print("site:",site)
            type.append('-->')

            if site != N:
                Dl, d_phys, Dr = G.shape
                R = newupdate(G.reshape(Dl, Dr * d_phys).T).T.reshape(Dl, d_phys, Dr)
                #print("tensorial derivative",G.transpose(1,0,2))
                #print("maximizer",R.transpose(1,0,2))
                
            else:
                Dl, d_phys = G.shape
                R = newupdate(G.T).T.reshape(Dl, d_phys, 1)
                #print("tensorial derivative SPIKE",G)
                #print("maximizer",R.transpose(1,0,2))
                #print(R.shape)

            psi.set_B(i, npc.Array.from_ndarray_trivial(R, labels=['vL', 'p', 'vR']))
            
            nstep += 1
            updates_since_check += 1
            k_curr = b.overlap(psi)
            currency.append(k_curr)
            steps.append(site)
            print("current price",k_curr)
            print("reference price",k_ref)

            if updates_since_check >= K:
                
                if np.abs(k_curr - k_ref) <= err * max(np.abs(k_ref), 1e-12):
                    return psi, k_curr, nstep, currency, steps, type
                k_ref = k_curr
                updates_since_check = 0
            
        #ORA SIAMO ARRIVATI A L*L*...A*R*...R* E DOBBIAMO FARE SWEEP RIGHT

        #OTTIMIZZIAMO DI NUOVO (è UNA DIVERSA OTTIMIZZAZIONE, NONTRIVIAL) FINO A L*L*..A*R**..R**R**
        for i in range(N-2, A_loc-1, -1):

            sweep = 'right'
            
            site = i + 1 ## quindi site va da N-1 A_loc+1

            G = tensorial_derivative(psi=psi, b=b, site=site).to_ndarray()
            #print("step:",nstep)
            #print("site:",site)
            type.append('<--')
            
            Dl, d_phys, Dr = G.shape
            R = newupdate(G.reshape(Dl, Dr * d_phys).T).T.reshape(Dl, d_phys, Dr)
            psi.set_B(i, npc.Array.from_ndarray_trivial(R, labels=['vL', 'p', 'vR']))

            nstep += 1
            updates_since_check += 1
            k_curr = b.overlap(psi)
            currency.append(k_curr)
            steps.append(site)
            print("current price",k_curr)
            print("reference price",k_ref)

            if updates_since_check >= K:
                
                if np.abs(k_curr - k_ref) <= err * max(np.abs(k_ref), 1e-12):
                    return psi, k_curr, nstep, currency, steps, type
                k_ref = k_curr
                updates_since_check = 0


        #SECONDO CONFLITTO
        psi, A_loc = double_conflict(b = b, psi = psi, A_loc = A_loc, direction = sweep)

        for i in range(A_loc-2, -1, -1):

            sweep = 'right'
            
            site = i + 1 ## quindi site va da Aloc-1 a 1 incluso


    raise RuntimeError("SweepingAlgorithm did not converge within max_steps")