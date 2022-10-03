
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyblock2.driver.core import SOCDMRGDriver, SymmetryTypes
import numpy as np
import pickle

import sys

assert len(sys.argv) == 3

fidx = 4
spins = [int(sys.argv[1])]
nroots = int(sys.argv[2])

driver = SOCDMRGDriver(
    scratch="/central/scratch/hczhai/soc-04-%02d-S%02d-NR%02d" % (fidx, spins[0], nroots),
    restart_dir="/central/scratch/hczhai/soc-mps-04-%02d-S%02d-NR%02d"% (fidx, spins[0], nroots),
    symm_type=SymmetryTypes.SU2, n_threads=28,
    stack_mem=int(50E9), mpi=True
)

driver.mpi.barrier()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, hso, idx = pickle.load(open("01-mf-data.bin", "rb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
txst = txed

for spin in spins:

    if driver.mpi.rank == driver.mpi.root:
        print('dmrg 2S = %d nroots = %d' % (spin, nroots))

    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)
    h1e[np.abs(h1e) < 1e-10] = 0
    g2e[np.abs(g2e) < 1e-10] = 0

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
    ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=nroots)

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

    driver.reorder_idx = idx

    skets = []
    for i in range(len(energies)):
        sket = driver.split_mps(ket, i, "S%d-KET%d" % (spin, i))
        if driver.mpi.rank == driver.mpi.root:
            print('split mps = ', i)
        skets.append(sket)

    txed = time.perf_counter()
    print("DMRG 2S=%d FINISH TIME = " % spin, datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    print("DMRG 2S=%d TIME  = %20.3f" % (spin, txed - txst))
    txst = txed

    np.save("%02d-S%02d-R%02d-energies.npy" % (fidx, spin, nroots), np.array(energies))
