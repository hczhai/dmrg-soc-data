
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyscf import scf, mcscf, gto
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2._pyscf.ao2mo import integrals as itg
import numpy as np
import pickle

fidx = 1

driver = DMRGDriver(
    scratch="/central/scratch/hczhai/soc-05-%02d" % fidx, symm_type=SymmetryTypes.SGFCPX, n_threads=28,
    stack_mem=int(40E9), mpi=True
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
            verbose=4, spin=5, charge=-3, max_memory=100000)
print('basis = dz nelec = %d nao = %d' % (mol.nelectron, mol.nao))

mf = scf.RHF(mol).x2c()

mf.mo_coeff, mf.mo_occ, mf.mo_energy, nactorb, nactelec, dmao = pickle.load(open("00-cas-data.bin", "rb"))

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

pickle.dump((ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, hso, idx), open("%02d-mf-data.bin" % fidx, "wb"))

txed = time.perf_counter()
print("MF FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("MF TIME  = %20.3f" % (txed - txst))
