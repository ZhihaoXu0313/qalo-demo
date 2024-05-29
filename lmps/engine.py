from lammps import lammps


def SNAPLammpsObj(lmp, infile):
    lmp.file(infile)

    natoms = lmp.get_natoms()
    types = lmp.gather_atoms("type", 0, 1)

    bs_array = lmp.extract_compute("SNA", 1, 2)
    return natoms, types, bs_array
