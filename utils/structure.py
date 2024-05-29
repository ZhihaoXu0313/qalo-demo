import numpy as np
import math

from arguments import JSONParameters
args = JSONParameters()


def idx2coord(idx, site_positions=args.structure.site, supercell=args.structure.sc):
    Nx, Ny, Nz = supercell[0], supercell[1], supercell[2]
    cellIndex, siteIndex = divmod(idx, len(site_positions))

    x_unit = cellIndex % Nx
    y_unit = (cellIndex // Nx) % Ny
    z_unit = cellIndex // (Nx * Ny)

    x_site, y_site, z_site = site_positions[siteIndex]
    x = (x_unit + x_site) / Nx
    y = (y_unit + y_site) / Ny
    z = (z_unit + z_site) / Nz

    return x, y, z


def coord2idx(x, y, z, site_positions=args.structure.site, supercell=args.structure.sc):
    Nx, Ny, Nz = supercell[0], supercell[1], supercell[2]
    x_unit = math.floor(x * Nx)
    y_unit = math.floor(y * Ny)
    z_unit = math.floor(z * Nz)

    unit_cell_index = x_unit + (y_unit * Nx) + (z_unit * Nx * Ny)
    try:
        site_index = site_positions.index([round(x * Nx, 2) % 1, round(y * Ny, 2) % 1, round(z * Nz, 2) % 1])
    except ValueError:
        raise Exception("Invalid site position")
    return int(unit_cell_index * len(site_positions) + site_index)


def distance(idx1, idx2, site_positions=args.structure.site, supercell=args.structure.sc):
    x1, y1, z1 = idx2coord(idx1, site_positions, supercell)
    x2, y2, z2 = idx2coord(idx2, site_positions, supercell)
    dx = abs(x1 - x2) if abs(x1 - x2) <= 0.5 else 1 - abs(x1 - x2)
    dy = abs(y1 - y2) if abs(y1 - y2) <= 0.5 else 1 - abs(y1 - y2)
    dz = abs(z1 - z2) if abs(z1 - z2) <= 0.5 else 1 - abs(z1 - z2)
    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return r



