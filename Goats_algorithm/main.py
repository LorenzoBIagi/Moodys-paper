import b_construction as B
import sweeping_algorithm as sweep
import plot_results as plt


def main():

    #PARAMETRI 

    #PARAMETRI ALBERO (SO: INITIAL ASSET PRICE, K: VALUE OF non lo so, N: LENGTH TENSOR TRAIN)
    S0, K, N = 100, 95, 10

    #COEFFICIENTI DA MOLTIPLICARE ASSET PRICE
    d_op, m_op, u_op = 0.9, 1.0, 1.1

    #PROBABILITà UP AND DOWN (mid è la normalizzazione)
    pd, pu = 0.25, 0.25

    #COSTRUISCO LA TT FISSATA DELLE B CON I PARAMETRI DELL'ALBERO
    B_list = B.build_B_list(S0, K, N, d_op, m_op, u_op, pd, pu)
    b_mps = B.build_mps(B_list, d=3)


    #APPLICO SWEEPING ALGORITHM ALL'ANSATZ PSI PER APPROSSIMARE COST FUNCTION 
    psi_out, price, nstep, currency, steps, types = sweep.Sweep_double_conflict(
        b=b_mps,
        d=3,
        D=10,
        err=1e-18,
        K=4
    )


    #RISULTATI
    print("Number of steps:", nstep)
    print("Final price:", price)

    plt.plot_sweeping_currency(steps, currency, types)


if __name__ == "__main__":
    main()