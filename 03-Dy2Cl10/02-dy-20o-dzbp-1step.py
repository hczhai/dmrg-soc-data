
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import numpy as np
import pickle

fidx = 2

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-04-%02d" % fidx,
    restart_dir="/central/scratch/hczhai/soc-mps-04-%02d" % fidx,
    symm_type=SymmetryTypes.SGFCPX, n_threads=56,
    stack_mem=int(200E9), mpi=True
)

driver.mpi.barrier()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = pickle.load(open("00-mf-data.bin", "rb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
txst = txed

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym, singlet_embedding=False)
h1e[np.abs(h1e) < 1e-10] = 0
g2e[np.abs(g2e) < 1e-10] = 0

mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore)
ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=16)
bond_dims = [500] * 4 + [1000] * 4 + [1500] * 4 + [2000] * 4
noises = [1e-4] * 4 + [1e-5] * 8 + [1e-6] * 4 + [0]
thrds = [1e-5] * 4 + [1e-6] * 8 + [5e-7]
energies = driver.dmrg(
    mpo,
    ket,
    n_sweeps=24,
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

driver.finalize()

txed = time.perf_counter()
print("DMRG FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("DMRG TIME  = %20.3f" % (txed - txst))
