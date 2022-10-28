
Data for ab initio Density Matrix Renormalization Group (DMRG) with Spin-Orbit Coupling (SOC)
=============================================================================================

This includes the input and output files for the following paper:

* Huanchen Zhai, and Garnet Kin-Lic Chan. "A comparison between the one-and two-step spin-orbit coupling approaches based on the ab initio Density Matrix Renormalization Group." J. Chem. Phys. **157**, 164108 (2022). doi: [10.1063/5.0107805](https://doi.org/10.1063/5.0107805).

Software dependences:

* ``python 3.8``
* DMRG: ``block2 >= 0.5.1`` (with mpi) [block2-preview](https://github.com/block-hczhai/block2-preview)
* DFT/CASSCF: ``pyscf 2.0.1`` [pyscf](https://github.com/pyscf/pyscf)

Files:

* Cu atom
  - [00] 1-step SOC approach
  - [02] 2-step SOC approach
* Au atom
  - [00] 1-step SOC approach
  - [02] 2-step SOC approach
* DyCl6 (monomer)
  - [00] mean-field and DMRG-CASSCF
  - [01] orbitals for 2-step (depends on [00])
  - [02] 1-step SOC approach (depends on [00])
  - [04] 2-step SOC approach with 21 states per multiplicity (depends on [01])
  - [05] 2-step SOC approach with 42 states per multiplicity (depends on [01])
  - [06] 2-step SOC approach with 84 states per multiplicity (depends on [01])
* Dy2Cl10 (dimer)
  - [00] mean-field and orbitals for 1-step
  - [01] orbitals for 2-step (depends on [00])
  - [02] 1-step SOC-DMRG (depends on [00])
  - [04] 2-step SU2-DMRG without SOC (depends on [01])
  - [05] 2-step SU2-DMRG 1TTDM (depends on [04])
  - [06] 2-step effective Hamiltonian diagonalization (depends on [05])
