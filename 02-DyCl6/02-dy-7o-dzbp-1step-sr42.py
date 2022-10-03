
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
import numpy as np
import pickle

fidx = 2

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-05-%02d" % fidx, symm_type=SymmetryTypes.SGFCPX, n_threads=28,
    stack_mem=int(40E9), mpi=True
)

driver.mpi.barrier()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, idx = pickle.load(open("00-mf-data.bin", "rb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
txst = txed

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym, singlet_embedding=False)
h1e[np.abs(h1e) < 1e-10] = 0
g2e[np.abs(g2e) < 1e-10] = 0

mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore)
ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=16)
bond_dims = [500] * 5 + [1000] * 5 + [1500] * 5
noises = [1e-4] * 5 + [1e-5] * 5 + [1e-6] * 5 + [0]
thrds = [1e-5] * 5 + [1e-5] * 5 + [1e-7] * 5 + [1E-8]
energies = driver.dmrg(
    mpo,
    ket,
    n_sweeps=20,
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=2,
    dav_max_iter=400,
    cutoff=1E-24
)

au2cm = 219474.63111558527
au2ev = 27.21139

e0 = energies[0]

if driver.mpi.rank == driver.mpi.root:
    for ix, ex in enumerate(energies):
        print("%5d %20.10f Ha %15.6f eV %10.4f cm-1" % (ix, ex, (ex - e0) * au2ev, (ex - e0) * au2cm))

txed = time.perf_counter()
print("DMRG FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("DMRG TIME  = %20.3f" % (txed - txst))
txst = txed

driver.reorder_idx = idx
ssq_mpo = driver.get_spin_square_mpo()

for i in range(len(energies)):
    sket = driver.split_mps(ket, i, "KET%d" % i)
    kmps = sket.deep_copy("X-TMP")
    ex = driver.expectation(kmps, mpo, kmps, iprint=0)
    sket.info.load_mutable()
    sket.load_mutable()
    kmps = sket.deep_copy("X-TMP")
    ssq = driver.expectation(kmps, ssq_mpo, kmps, iprint=0)
    if driver.mpi.rank == driver.mpi.root:
        print('SPLIT MPS = %5d :: energy = %15.8f + %15.8f i <S^2> = %15.8f + %15.8f i'
              % (i, ex.real, ex.imag, ssq.real, ssq.imag))

txed = time.perf_counter()
print("SSQ FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("SSQ TIME  = %20.3f" % (txed - txst))
