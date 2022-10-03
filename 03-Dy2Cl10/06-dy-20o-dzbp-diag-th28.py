
import time
from datetime import datetime
txst = time.perf_counter()
print("START TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

from pyblock2.driver.core import SOCDMRGDriver, SymmetryTypes
import numpy as np
import pickle
import sys

fidx = 6

assert len(sys.argv) == 2

nroots = int(sys.argv[1])

driver = SOCDMRGDriver(
    scratch="/central/scratch/hczhai/soc-04-%02d" % fidx,
    symm_type=SymmetryTypes.SU2, n_threads=28,
    stack_mem=int(10E9), mpi=True
)

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym, hso, idx = pickle.load(open("01-mf-data.bin", "rb"))
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)

all_eners = pickle.load(open("%02d-S%02d-NR%02d-1pdm-data.bin" % (5, 0, nroots), "rb"))[0]

twoss = [0, 2, 4, 6, 8, 10]
pdms_dict = {}
for si in range(len(twoss)):
    dd = pickle.load(open("%02d-S%02d-NR%02d-1pdm-data.bin" % (5, twoss[si], nroots), "rb"))[1]
    pdms_dict.update(dd)

for k, v in pdms_dict.items():
    print(k, np.linalg.norm(v))

print(all_eners, len(twoss) , pdms_dict.keys())

energies = driver.soc_two_step(all_eners, twoss, pdms_dict, hso, iprint=1)

au2cm = 219474.63111558527
au2ev = 27.21139

e0 = energies[0]

if driver.mpi.rank == driver.mpi.root:
    for ix, ex in enumerate(energies):
        print("%5d %20.10f Ha %15.6f eV %10.4f cm-1" % (ix, ex, (ex - e0) * au2ev, (ex - e0) * au2cm))

driver.finalize()

txed = time.perf_counter()
print("DIAG FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("DIAG TIME  = %20.3f" % (txed - txst))
