
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import numpy as np
import pickle

fidx = 2

load_dir = "/central/scratch/hczhai/soc-mps-04-%02d" % fidx

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-04-%02d-proj-ssq" % fidx,
    symm_type=SymmetryTypes.SGFCPX, n_threads=28,
    stack_mem=int(100E9), mpi=True
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

if driver.mpi is not None:
    driver.mpi.barrier()
if driver.mpi.rank == driver.mpi.root:
    import shutil
    import os
    for k in os.listdir(load_dir):
        if '.KET.' in k or k == 'KET-mps_info.bin':
            shutil.copy(load_dir + "/" + k, driver.scratch + "/" + k)
if driver.mpi is not None:
    driver.mpi.barrier()

mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore)
ket = driver.load_mps(tag="KET", nroots=16)

driver.reorder_idx = np.array([
    33, 31, 19, 25, 23, 21,  3, 29,  7, 27,  9,  5, 39, 13, 37,
    15,  1, 17, 11, 35, 34, 10, 16,  0, 14, 36, 12, 38,  4,  8,
    26,  6, 28,  2, 20, 22, 24, 18, 30, 32], dtype=int)
ssq_mpo = driver.get_spin_square_mpo()

au2cm = 219474.63111558527
au2ev = 27.21139

energies = []
for i in range(16):
    if driver.mpi.rank == driver.mpi.root:
        print('split mps = ', i)
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
        n_sweeps=6,
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
    ssq = driver.expectation(sket, ssq_mpo, sket, iprint=2)
    pket = sket.deep_copy("S%d-KET%d" % (spin, i))
    pket.info.save_data(driver.scratch + "/%s-mps_info.bin" % pket.info.tag)
    driver.mpi.barrier()
    if driver.mpi.rank == driver.mpi.root:
        print('SPLIT MPS = %5d :: energy = %15.8f + %15.8f i <S^2> = %15.8f + %15.8f i'
              % (i, ex.real, ex.imag, ssq.real, ssq.imag))
    energies.append(ex.real)

e0 = energies[0]

if driver.mpi.rank == driver.mpi.root:
    for ix, ex in enumerate(energies):
        print("%5d %20.10f Ha %15.6f eV %10.4f cm-1" % (ix, ex, (ex - e0) * au2ev, (ex - e0) * au2cm))

txed = time.perf_counter()
print("SSQ FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("SSQ TIME  = %20.3f" % (txed - txst))
