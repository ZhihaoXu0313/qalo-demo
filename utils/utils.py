import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

import numpy as np
from itertools import combinations
from quaternary import quaternary
import sys
import ctypes
from contextlib import contextmanager

from pymatgen.io.vasp import Poscar

from arguments import JSONParameters
args = JSONParameters()


def plot2dstruct(data, fpath):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = len(args.structure.site) * Nx * Ny * Nz

    fig, ax = plt.subplots()
    plt.imshow(data)
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.4)
    ax.set_xlabel("sites")
    ax.set_ylabel("species")
    ax.set_xticks(np.arange(0, nsites, step=10))
    ax.set_yticks(np.arange(0, len(args.basic['elements']), step=1))
    plt.colorbar()
    plt.savefig(fpath, dpi=300)


def barycentric2cartesian(b_array):
    verts = [[0, 0, 0],
             [1, 0, 0],
             [0.5, np.sqrt(3) / 2, 0],
             [0.5, 0.28867513, 0.81649658]]
    t = np.transpose(np.array(verts))
    t_array = np.array([t.dot(x) for x in b_array])
    return t_array


def plotQuaternaryDiagrams(df, path_of_out=args.work_path.output):
    label = df['energy']
    phase = df.iloc[:, 1:] * 100

    fig = plt.figure()
    quat = quaternary(fig)
    quat.set_grid()
    quat.set_label1('Nb')
    quat.set_label2('Mo')
    quat.set_label3('Ta')
    quat.set_label4('W', pad=0.05)
    for i in range(len(label)):
        quadia = quat.scatter(phase.iloc[i]['Nb'],
                              phase.iloc[i]['Mo'],
                              phase.iloc[i]['Ta'],
                              marker='o', c=label.iloc[i],
                              norm=plt.Normalize(min(label.values), max(label.values)))
    fig.colorbar(quadia)
    # fig.show()
    fig.savefig(os.path.join(path_of_out, "quaternary_diagram.png"), dpi=300)


@contextmanager
def redirect(target):
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())

    with open(target, 'a') as log_file:
        log_fd = log_file.fileno()
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())

        try:
            yield
        finally:
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
            

def calculate_pcf(poscar, r_min=2.4, r_max=22.8, dr=0.1):
    structure = Poscar.from_file(poscar).structure
    n_bins = int(r_max / dr)
    n_pairs = int((1 + len(args.basic.elements)) * len(args.basic.elements) * 0.5) + 1
    
    pcf = np.zeros((n_bins, 1 + n_pairs))
    r_bins = np.arange(0, r_max + dr, dr)
    pcf[:, 0] = r_bins[:-1] + dr / 2
    
    pair_list = [(e1, e2) for i, e1 in enumerate(args.basic.elements) for e2 in args.basic.elements[i:]]
    pair_list.append(("random", "random"))
    
    for p, (element1, element2) in enumerate(pair_list):
        distance = []
        for i, site1 in enumerate(structure):
            if site1.specie.symbol == element1 or element1 == "random":
                for j, site2 in enumerate(structure):
                    if site2.specie.symbol == element2 or element2 == "random":
                        if i != j:
                            d = structure.get_distance(i, j)
                            if d <= r_max:
                                distance.append(d)
        gr, _ = np.histogram(distance, bins=r_bins, density=True)
        pcf[:, p + 1] = gr
    
    return pcf
        
    