
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import scf, mcscf, gto
from pyblock2._pyscf import scf as b2scf
from pyblock2._pyscf import mcscf as b2mcscf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
import numpy as np
import pickle

fidx = 0

b = 2.72

coords = [
    ['Dy', [0, 0, 0]],
    ['Dy', [b, b, 0]],
    ['Cl', [-b, 0, 0]],
    ['Cl', [b, 0, 0]],
    ['Cl', [0, -b,  0]],
    ['Cl', [0, b,  0]],
    ['Cl', [0, 0, -b]],
    ['Cl', [0, 0, b]],
    ['Cl', [b, 2 * b, 0]],
    ['Cl', [2 * b, b, 0]],
    ['Cl', [b, b, -b]],
    ['Cl', [b, b, b]]
]

mol = gto.M(atom=coords, basis={"Dy": "ano@7s6p4d2f", "Cl": "ano@4s3p"},
            verbose=4, spin=10, charge=-4, max_memory=100000)
print('basis = dz nelec = %d nao = %d' % (mol.nelectron, mol.nao))

mf = scf.UKS(mol).x2c()
mf = b2scf.smearing_(mf, sigma=0.2, method="fermi", fit_spin=True)
mf.xc = "bp86"
mf.conv_tol = 1e-10
mf.max_cycle = 500
mf.diis_space = 15

dm0 = b2scf.get_metal_init_guess(mol, orb="4f", atom_idxs=[0, 1], coupling="++", atomic_spin=5)
mf.kernel(dm0=dm0)
dmao = np.einsum('yij->ij', mf.make_rdm1(), optimize=True)

lo_coeff, lo_occ, lo_energy = b2mcscf.get_uno(mf)
selected = b2mcscf.select_active_space(
    mol, lo_coeff, lo_occ, ao_labels=['Dy-4f', '3-Cl-3p', '5-Cl-3p'],
    atom_order=[0, 3, 5, 1]
)
lo_coeff, lo_occ, lo_energy, nactorb, nactelec = b2mcscf.sort_orbitals(
    mol,
    lo_coeff,
    lo_occ,
    lo_energy,
    cas_list=selected,
    do_loc=True,
    split_low=0.1,
    split_high=1.9,
)

b2scf.mulliken_pop_dmao(mol, mf.make_rdm1())

pickle.dump((lo_coeff, lo_occ, lo_energy, nactorb, nactelec, dmao), open("%02d-cas-data.bin" % fidx, "wb"))

mf = scf.RHF(mol).x2c()
mf.mo_coeff = lo_coeff
mf.mo_occ = np.array([int(np.round(x) + 0.1) for x in lo_occ])
assert sum(mf.mo_occ) == mol.nelectron
mf.mo_energy = lo_energy

mc = mcscf.CASCI(mf, nactorb, nactelec)
ncore = mc.ncore
ncas = mc.ncas
print('ncore = ', ncore, ' ncas = ', ncas)

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_rhf_somf_integrals(
    mf, ncore, ncas, pg_symm=False, dmao=dmao, amfi=True, x2c1e=False, x2c2e=False
)

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-04-%02d" % fidx,
    symm_type=SymmetryTypes.SGFCPX, n_threads=56,
    stack_mem=int(70E9)
)

idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
print('reordering = ', idx)
h1e = h1e[idx][:, idx]
g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

shift = -28915.30
x = (shift - ecore) / n_elec
h1e -= x * np.identity(len(h1e))
ecore += x * n_elec

pickle.dump((ncas, n_elec, spin, ecore, h1e, g2e, orb_sym), open("%02d-mf-data.bin" % fidx, "wb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
