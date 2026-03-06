from b_construction import build_B_list, build_mps
from sweeping_algorithm import SweepingAlgorithm
from plot_results import plot_sweeping_currency


def main():
    S0, K, N = 100, 95, 10
    d_op, m_op, u_op = 0.9, 1.0, 1.1
    pd, pu = 0.25, 0.25

    B_list = build_B_list(S0, K, N, d_op, m_op, u_op, pd, pu)
    b_mps = build_mps(B_list, d=3)

    psi_out, price, nstep, currency, steps, types = SweepingAlgorithm(
        b=b_mps,
        d=3,
        D=10,
        err=1e-18
    )

    print("Number of steps:", nstep)
    print("Final price:", price)

    plot_sweeping_currency(steps, currency, types)


if __name__ == "__main__":
    main()