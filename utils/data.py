import numpy as np
import ase
from pymatgen.core import Structure
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import re
import csv
import pickle

from utils.structure import coord2idx, idx2coord
from arguments import JSONParameters
args = JSONParameters()


def extract_composition(poscar):
    structure = Structure.from_file(poscar)
    composition = structure.composition
    composition_list = []
    for element, count in composition.get_el_amt_dict().items():
        composition_list.append(count)
    return composition_list


def extract_toten(outcar):
    if not os.path.exists(outcar):
        raise FileNotFoundError(f"File not found: {outcar}")
    with open(outcar, 'r') as file:
        content = file.read()

    matches = re.findall(r'TOTEN\s*=\s*(-?\d+\.\d+)', content)
    if matches:
        return matches[-1]
    else:
        return None
    

def poscar2binvec(poscar, spcOriginal, spcDesign=args.structure.sc, flatten=False):
    structure = Structure.from_file(poscar)
    scaling_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if spcOriginal != spcDesign:
        scaling_matrix = [[int(spcDesign[0] // spcOriginal[0]), 0, 0],
                          [0, int(spcDesign[1] // spcOriginal[1]), 0],
                          [0, 0, int(spcDesign[2] // spcOriginal[2])]]
        structure.make_supercell(scaling_matrix)
    element_list = list(dict.fromkeys([s.species_string for s in structure]))
    binvec = np.zeros((len(element_list), len(structure)), dtype=int)
    for index, site in enumerate(structure):
        element = site.species_string
        coords = site.frac_coords
        siteid = coord2idx(coords[0], coords[1], coords[2])
        binvec[element_list.index(element), siteid] = 1
    if flatten:
        binvec = binvec.flatten()
    return binvec, scaling_matrix


def binvec2poscar(binvec, poscar):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nspecies = len(args.basic.elements)
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz
    
    system = {}
    system['comment'] = "binary vector to poscar structure by QALO"
    system['scale_coeff'] = float(1.0)
    system['box_coord'] = []
    system['box_coord'].append([Nx * args.structure.alat, 0.0, 0.0])
    system['box_coord'].append([0.0, Ny * args.structure.alat, 0.0])
    system['box_coord'].append([0.0, 0.0, Nz * args.structure.alat])
    system['atom'] = args.basic.elements

    compositions = []
    for i in range(nspecies):
        compositions.append(int(sum(binvec[i * nsites: (i + 1) * nsites])))

    system['atom_num'] = compositions
    system['all_atom'] = sum(system['atom_num'])
    system['coord_type'] = 'Direct'
    system['coord_Direct'] = []
    indices = np.where(binvec == 1)[0]
    for a in range(int(system['all_atom'])):
        i, j = indices[a] // nsites, indices[a] % nsites
        x, y, z = idx2coord(j)
        system['coord_Direct'].append([x, y, z])

    with open(poscar, 'w') as f:
        f.writelines(" " + system['comment'] + "\n")
        f.writelines("   " + str(system['scale_coeff']) + "\n")
        for i in range(3):
            f.writelines("     " + ' '.join(str('%.16f' % x) for x in system['box_coord'][i]) + "\n")
        f.writelines(" " + ' '.join(str(x) for x in system['atom']) + "\n")
        f.writelines("  " + ' '.join(str(x) for x in system['atom_num']) + "\n")
        f.writelines(system['coord_type'] + "\n")
        for i in range(int(system['all_atom'])):
            if system['coord_type'] == 'Direct':
                f.writelines("   " + ' '.join(str('%.16f' % x) for x in system['coord_Direct'][i]) + "\n")
    return system


def raw2libffm(filepath, spcOriginal):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz

    entries = os.listdir(filepath)
    random.shuffle(entries)
    libffm_data = []
    for structure in entries:
        d = os.path.join(filepath, structure)
        if os.path.isdir(d) and structure.startswith("s-"):
            binvec, scaling_matrix = poscar2binvec(os.path.join(d, "POSCAR"), spcOriginal=spcOriginal, flatten=True)
            toten = (float(extract_toten(os.path.join(d, "OUTCAR"))) *
                     scaling_matrix[0][0] * scaling_matrix[1][1] * scaling_matrix[2][2])
            libffm_row = [str(toten)]
            for i, value in enumerate(binvec):
                field = i % nsites
                feature = (i // nsites) * nsites + field
                libffm_row.append(f"{field}:{feature}:{int(value)}")
            libffm_data.append(' '.join(libffm_row))
    return libffm_data


def write_libffm_data(datapath, fmpath):
    datasets = os.listdir(datapath)
    libffm_data = []
    for ds in datasets:
        dims = ds.split('-')[-1]
        dimxyz = dims.split('x')
        spcOriginal = [int(dimxyz[0]), int(dimxyz[1]), int(dimxyz[2])]
        d = os.path.join(datapath, ds)
        libffm_data.extend(raw2libffm(d, spcOriginal))
    random.shuffle(libffm_data)
    with open(os.path.join(fmpath, "libffm_data_total.txt"), 'a') as file:
        for line in libffm_data:
            file.write(line + '\n')
    return libffm_data


def add_new_structure(energy, dfstack, fmpath, newpath):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz

    dfstack.insert(loc=0, column='energy', value=energy)
    new_libffm_data = []
    new_binvec_structure = []
    for index, row in dfstack.iterrows():
        label = row['energy']
        features = row.drop('energy')
        libffm_row = [str(label)]
        binvec_row = []
        for i, value in enumerate(features):
            field = i % nsites
            feature = (i // nsites) * nsites + field
            libffm_row.append(f"{field}:{feature}:{int(value)}")
            binvec_row.append(str(int(value)))
        new_libffm_data.append(' '.join(libffm_row))
        new_binvec_structure.append(' '.join(binvec_row))
    with open(os.path.join(fmpath, "libffm_data_total.txt"), 'a') as file:
        for line in new_libffm_data:
            file.write(line + '\n')
    with open(os.path.join(newpath, "new_structure.txt"), 'a') as file:
        for line in new_binvec_structure:
            file.write(line + '\n')
    return new_libffm_data, new_binvec_structure


def generate_ffm_data(fmpath, sample_ratio):
    with open(os.path.join(fmpath, "libffm_data_total.txt"), 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    n_pick = int(len(lines) * sample_ratio)
    select_data = lines[:n_pick]
    with open(os.path.join(fmpath, "train_ffm.txt"), "w") as file:
        file.writelines(select_data[:int(0.8 * n_pick)])
    with open(os.path.join(fmpath, "valid_ffm.txt"), "w") as file:
        file.writelines(select_data[int(0.8 * n_pick):int(0.9 * n_pick)])
    with open(os.path.join(fmpath, "test_ffm.txt"), "w") as file:
        file.writelines(select_data[int(0.9 * n_pick):])


def save_energy(filepath, energy, composition):
    if not isinstance(energy, (list, tuple)):
        raise ValueError("Energy must be a list or tuple")
    with open(filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(energy + composition)
