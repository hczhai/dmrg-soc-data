
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyblock2.driver.core import SOCDMRGDriver, SymmetryTypes
import numpy as np
import pickle

import sys

assert len(sys.argv) == 3

fidx = 5
gspin = int(sys.argv[1])
nroots = int(sys.argv[2])

orig_dir = '/central/scratch/hczhai/soc-04-%02d-SXX-NR%02d' % (4, nroots)

driver = SOCDMRGDriver(
    scratch="/central/scratch/hczhai/soc-04-%02d-S%02d-NR%02d-proj" % (fidx, gspin, nroots),
    symm_type=SymmetryTypes.SU2, n_threads=28,
    stack_mem=int(50E9), mpi=True
)

driver.mpi.barrier()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, hso, idx = pickle.load(open("01-mf-data.bin", "rb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
txst = txed

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)
h1e[np.abs(h1e) < 1e-10] = 0
g2e[np.abs(g2e) < 1e-10] = 0

mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)

twoss = [gspin]
for si in range(len(twoss)):
    spin = twoss[si]

    if driver.mpi is not None:
        driver.mpi.barrier()
    if driver.mpi.rank == driver.mpi.root:
        import shutil
        import os
        xdir = orig_dir.replace("SXX", "S%02d" % twoss[si])
        for k in os.listdir(xdir):
            if '.KET.' in k or k == 'KET-mps_info.bin':
                shutil.copy(xdir + "/" + k, driver.scratch + "/" + k)
    if driver.mpi is not None:
        driver.mpi.barrier()

    ket = driver.load_mps("KET", nroots=nroots)

    mpss = []
    xeners = []
    for i in range(nroots):
        if driver.mpi.rank == driver.mpi.root:
            print('S%02d - split mps = ' % twoss[si], i)
        sket = driver.split_mps(ket, i, "X-KET%d" % i)
        proj_mpss = []
        for j in range(i):
            driver.mpi.barrier()
            pjmps = driver.load_mps("S%d-KET%d" % (spin, j))
            driver.mpi.barrier()
            proj_mpss.append(pjmps.deep_copy("X-KET%d" % j))
            driver.mpi.barrier()
        bond_dims = [2000] * 4
        noises = [1e-5] * 2 + [0]
        thrds = [5e-7] * 4
        ex = driver.dmrg(
            mpo,
            sket,
            n_sweeps=4,
            tol=0,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=2,
            dav_max_iter=400,
            cutoff=1E-24,
            proj_mpss=proj_mpss,
            proj_weights=[0.5] * len(proj_mpss),
        )
        pket = sket.deep_copy("S%d-KET%d" % (spin, i))
        pket.info.save_data(driver.scratch + "/%s-mps_info.bin" % pket.info.tag)
        driver.mpi.barrier()
        if driver.mpi.rank == driver.mpi.root:
            print('S%02d - energy = ' % twoss[si], ex)
        xeners.append(ex)

if driver.mpi.rank == driver.mpi.root:
    pickle.dump(xeners, open("%02d-S%02d-NR%02d-proj-ener-data.bin" % (fidx, gspin, nroots), "wb"))

txed = time.perf_counter()
print("PROJ FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("PROJ TIME  = %20.3f" % (txed - txst))
