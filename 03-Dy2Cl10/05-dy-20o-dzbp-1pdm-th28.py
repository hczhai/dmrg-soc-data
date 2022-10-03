
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
    scratch="/central/scratch/hczhai/soc-04-%02d-S%02d-NR%02d" % (fidx, gspin, nroots),
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

twoss = [0, 2, 4, 6, 8, 10]
all_mpss = []
all_eners = []
for si in range(len(twoss)):

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
        sket = driver.split_mps(ket, i, "S%d-KET%d" % (twoss[si], i))
        if driver.mpi.rank == driver.mpi.root:
            print('S%02d - split mps = ' % twoss[si], i)
        if gspin == 0:
            kmps = sket.deep_copy("X-TMP")
            ex = driver.expectation(kmps, mpo, kmps, iprint=2)
            if driver.mpi.rank == driver.mpi.root:
                print('S%02d - energy = ' % twoss[si], ex)
            xeners.append(ex)
        mpss.append(sket)

    all_eners.append(xeners)
    all_mpss.append(mpss)

txed = time.perf_counter()
print("ENER FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("ENER TIME  = %20.3f" % (txed - txst))
txst = txed

driver.reorder_idx = idx

pdms_dict = {}
ip = 0
for si in range(len(twoss)):
    for i, iket in enumerate(all_mpss[si]):
        if twoss[si] != gspin:
            continue
        jp = 0
        for sj in range(len(twoss)):
            for j, jket in enumerate(all_mpss[sj]):
                if i + ip < j + jp or abs(twoss[si] - twoss[sj]) > 2:
                    continue
                if driver.mpi.rank == driver.mpi.root:
                    print('compute pdm = ', i + ip, j + jp)
                dm = driver.get_trans_1pdm(iket, jket, soc=True, iprint=2)
                if driver.mpi.rank == driver.mpi.root:
                    print('pdm = ', np.linalg.norm(dm))
                pdms_dict[(i + ip, j + jp)] = dm
            jp += len(all_mpss[sj])
    ip += len(all_mpss[si])

if driver.mpi.rank == driver.mpi.root:
    pickle.dump((all_eners, pdms_dict), open("%02d-S%02d-NR%02d-1pdm-data.bin" % (fidx, gspin, nroots), "wb"))

txed = time.perf_counter()
print("PDM FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("PDM TIME  = %20.3f" % (txed - txst))
