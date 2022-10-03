
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import scf, mcscf, gto, dmrgscf, lib
from pyblock2._pyscf import scf as b2scf
from pyblock2._pyscf import mcscf as b2mcscf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
import numpy as np
import pickle
import os

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

fidx = 0

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-05-%02d" % fidx, symm_type=SymmetryTypes.SGFCPX, n_threads=28,
    stack_mem=int(40E9), mpi=False
)

b = 2.72

coords = [
    ["Dy", [0, 0, 0]],
    ["Cl", [-b, 0, 0]],
    ["Cl", [b, 0, 0]],
    ["Cl", [0, -b, 0]],
    ["Cl", [0, b, 0]],
    ["Cl", [0, 0, -b]],
    ["Cl", [0, 0, b]],
]

mol = gto.M(atom=coords, basis={"Dy": "ano@7s6p4d2f", "Cl": "ano@4s3p"},
            verbose=4, spin=5, charge=-3, max_memory=20000)
print('basis = dz nelec = %d nao = %d' % (mol.nelectron, mol.nao))

mf = scf.UKS(mol).x2c()
mf = b2scf.smearing_(mf, sigma=0.2, method="fermi", fit_spin=True)
mf.xc = "bp86"
mf.conv_tol = 1e-10
mf.max_cycle = 500
mf.diis_space = 15

dm0 = b2scf.get_metal_init_guess(mol, orb="4f", atom_idxs=[0], coupling="+", atomic_spin=5)
mf.kernel(dm0=dm0)
dmao = np.einsum('yij->ij', mf.make_rdm1(), optimize=True)

lo_coeff, lo_occ, lo_energy = b2mcscf.get_uno(mf)
selected = b2mcscf.select_active_space(
    mol, lo_coeff, lo_occ, ao_labels=["Dy-4f"], atom_order=[0]
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

mf = scf.RHF(mol).x2c()
mf.mo_coeff = lo_coeff
mf.mo_occ = np.array([int(np.round(x) + 0.1) for x in lo_occ])
assert sum(mf.mo_occ) == mol.nelectron
mf.mo_energy = lo_energy

lib.param.TMPDIR = os.path.abspath(lib.param.TMPDIR)

mc = mcscf.CASSCF(mf, nactorb, nactelec)
mcfs = [None] * 3
for i in range(3):
    mcfs[i] = dmrgscf.DMRGCI(mol, maxM=1500, tol=1E-10)
    mcfs[i].spin = i * 2 + 1
    mcfs[i].nroots = 42 if i != 2 else 21
    mcfs[i].runtimeDir = lib.param.TMPDIR + "/%d" % i
    mcfs[i].scratchDirectory = lib.param.TMPDIR + "/%d" % i
    mcfs[i].threads = 16
    mcfs[i].memory = int(mol.max_memory / 1000) # mem in GB
    mcfs[i].block_extra_keyword = ["real_density_matrix", "davidson_soft_max_iter 1600", "noreorder", "cutoff 1E-24"]
mc = mcscf.addons.state_average_mix(mc, mcfs, np.ones(42 + 42 + 21) / (42.0 + 42.0 + 21.0))
mc.kernel()
mf.mo_coeff = mc.mo_coeff

pickle.dump((mf.mo_coeff, mf.mo_occ, mf.mo_energy, nactorb, nactelec, dmao), open("%02d-cas-data.bin" % fidx, "wb"))

ncore = mc.ncore
ncas = mc.ncas
print('ncore = ', ncore, ' ncas = ', ncas)

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_rhf_somf_integrals(
    mf, ncore, ncas, pg_symm=False, dmao=dmao, amfi=True, x2c1e=False, x2c2e=False
)

idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
print('reordering = ', idx)
h1e = h1e[idx][:, idx]
g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

pickle.dump((ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, idx), open("%02d-mf-data.bin" % fidx, "wb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
