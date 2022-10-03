
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import scf, mcscf, gto
from pyblock2.driver.core import SOCDMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2._pyscf.ao2mo import integrals as itg
import numpy as np
import pickle

import sys

fidx = 1

driver = SOCDMRGDriver(
    scratch="/scratch/global/hczhai/soc-04-%02d" % fidx,
    symm_type=SymmetryTypes.SU2, n_threads=56,
    stack_mem=int(200E9), mpi=True
)

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

lo_coeff, lo_occ, lo_energy, nactorb, nactelec, dmao = pickle.load(open("00-cas-data.bin", "rb"))

mf = scf.RHF(mol).x2c()
mf.mo_coeff = lo_coeff
mf.mo_occ = np.array([int(np.round(x) + 0.1) for x in lo_occ])
assert sum(mf.mo_occ) == mol.nelectron
mf.mo_energy = lo_energy

mc = mcscf.CASCI(mf, nactorb, nactelec)
ncore = mc.ncore
ncas = mc.ncas
print('ncore = ', ncore, ' ncas = ', ncas)

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, ncore, ncas, pg_symm=False)
hsoao = itgsoc.get_somf_hsoao(mf, dmao=dmao, amfi=True, x2c1e=False, x2c2e=False)
hso = np.einsum('rij,ip,jq->rpq', hsoao,
    mf.mo_coeff[:, ncore:ncore + ncas],
    mf.mo_coeff[:, ncore:ncore + ncas], optimize=True)

idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
print('reordering = ', idx)
h1e = h1e[idx][:, idx]
g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

shift = -28915.30
x = (shift - ecore) / n_elec
h1e -= x * np.identity(len(h1e))
ecore += x * n_elec

pickle.dump((ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, hso, idx), open("%02d-mf-data.bin" % fidx, "wb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
