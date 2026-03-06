from psi_construction import build_mps_LAR

psi = build_mps_LAR(d=3,D=3,N=5)

for i in range (psi.L):
    if i == 0 or i == psi.L:
        print(psi.get_B(i).to_ndarray().shape)
        print(psi.get_B(i).to_ndarray().transpose(1,0,2))
    else:
        print(psi.get_B(i).to_ndarray().shape)
        print(psi.get_B(i).to_ndarray().transpose(1,0,2))