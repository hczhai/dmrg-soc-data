
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import scf, mcscf, gto
from pyscf.data import nist
from pyblock2._pyscf import scf as b2scf
from pyblock2._pyscf import mcscf as b2mcscf
from pyblock2.driver.core import SOCDMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2._pyscf.ao2mo import integrals as itg
import numpy as np
import pickle

fidx = 4

driver = SOCDMRGDriver(
    scratch="/central/scratch/hczhai/soc-05-%02d" % fidx, symm_type=SymmetryTypes.SU2, n_threads=28,
    stack_mem=int(40E9), mpi=True
)

driver.mpi.barrier()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, hso, idx = pickle.load(open("01-mf-data.bin", "rb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
txst = txed

nroots = 21
all_eners = []
all_mpss = []
twoss = []

for spin in [1, 3, 5]:

    if driver.mpi.rank == driver.mpi.root:
        print('dmrg 2S = %d nroots = %d' % (spin, nroots))

    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)
    h1e[np.abs(h1e) < 1e-10] = 0
    g2e[np.abs(g2e) < 1e-10] = 0

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
    ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=nroots)

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

    all_eners.append(energies)
    twoss.append(spin)
    all_mpss.append(skets)

if driver.mpi.rank == driver.mpi.root:
    print('total number of mpss = ', sum([len(x) for x in all_eners]))

pdms_dict = {}
ip = 0
for si in range(len(twoss)):
    for i, iket in enumerate(all_mpss[si]):
        jp = 0
        for sj in range(len(twoss)):
            for j, jket in enumerate(all_mpss[sj]):
                if i + ip < j + jp or abs(twoss[si] - twoss[sj]) > 2:
                    continue
                if driver.mpi.rank == driver.mpi.root:
                    print('compute pdm = ', i + ip, j + jp)
                dm = driver.get_trans_1pdm(iket, jket, soc=True, iprint=2)
                pdms_dict[(i + ip, j + jp)] = dm
            jp += len(all_eners[sj])
    ip += len(all_eners[si])

txed = time.perf_counter()
print("1PDM FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("1PDM TIME  = %20.3f" % (txed - txst))
txst = txed

pickle.dump(all_eners, open("%02d-sf-energies.bin" % fidx, "wb"))

energies = driver.soc_two_step(all_eners, twoss, pdms_dict, hso, iprint=1)

au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
au2ev = 27.21139

e0 = energies[0]

if driver.mpi.rank == driver.mpi.root:
    for ix, ex in enumerate(energies):
        print("%5d %20.10f Ha %15.6f eV %10.4f cm-1" % (ix, ex, (ex - e0) * au2ev, (ex - e0) * au2cm))

driver.finalize()

txed = time.perf_counter()
print("DIAG FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("DIAG TIME  = %20.3f" % (txed - txst))
