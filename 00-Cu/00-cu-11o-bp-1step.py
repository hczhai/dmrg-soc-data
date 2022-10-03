
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import scf, mcscf, gto
from pyscf.data import nist
from pyblock2._pyscf import mcscf as b2mcscf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
import numpy as np
import pickle

fidx = 0

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-00-%02d" % fidx, symm_type=SymmetryTypes.SGFCPX, n_threads=24,
    stack_mem=int(40E9), mpi=True
)

if driver.mpi.rank == driver.mpi.root:

    mol = gto.M(atom="Cu 0 0 0", basis="ano@6s5p3d2f", verbose=4, spin=1, max_memory=100000)
    print('basis = ano@6s5p3d2f nelec = %d nao = %d' % (mol.nelectron, mol.nao))
    mf = scf.RHF(mol).x2c().run(conv_tol=1e-14)
    dmao = np.einsum('yij->ij', mf.make_rdm1(), optimize=True)

    selected = b2mcscf.select_active_space(
        mol, mf.mo_coeff, mf.mo_occ, ao_labels=['Cu-3d', 'Cu-4d', 'Cu-4s'], atom_order=[0]
    )
    mf.mo_coeff, mf.mo_occ, mf.mo_energy, ncas, ncaselec = b2mcscf.sort_orbitals(
        mol,
        mf.mo_coeff,
        mf.mo_occ,
        mf.mo_energy,
        cas_list=selected,
        do_loc=True,
        split_low=0.1,
        split_high=1.9,
    )

    mc = mcscf.CASSCF(mf, ncas, ncaselec).state_average_((0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    mc.kernel()
    mf.mo_coeff = mc.mo_coeff

    ncore = mc.ncore
    ncas = mc.ncas
    print('ncore = ', ncore, ' ncas = ', ncas)

    np.save('%02d-mo_coeff.npy' % fidx, mf.mo_coeff)

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_rhf_somf_integrals(
        mf, ncore, ncas, pg_symm=False, dmao=dmao, amfi=True, x2c1e=False, x2c2e=False
    )

    idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
    print('reordering = ', idx)
    h1e = h1e[idx][:, idx]
    g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

    pickle.dump((ncas, n_elec, spin, ecore, h1e, g2e, orb_sym), open("%02d-mf-data.bin" % fidx, "wb"))

driver.mpi.barrier()
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = pickle.load(open("%02d-mf-data.bin" % fidx, "rb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
txst = txed

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym, singlet_embedding=False)
h1e[np.abs(h1e) < 1e-10] = 0
g2e[np.abs(g2e) < 1e-10] = 0

mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore)
ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=12)
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

au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2

e0 = np.average(energies[0:2])
e1 = np.average(energies[2:8])
e2 = np.average(energies[8:12])

au2ev = 27.21139

if driver.mpi.rank == driver.mpi.root:
    print("")
    print("E 2D(5/2)         = %10.4f eV" % ((e1 - e0) * au2ev))
    print("E 2D(3/2)         = %10.4f eV" % ((e2 - e0) * au2ev))
    print("2D(5/2) - 2D(3/2) = %10.4f eV" % ((e2 - e1) * au2ev))

driver.finalize()

txed = time.perf_counter()
print("DMRG FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("DMRG TIME  = %20.3f" % (txed - txst))
