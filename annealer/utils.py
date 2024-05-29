import numpy as np 
import re 
import pandas as pd 
import os 

from arguments import JSONParameters
args = JSONParameters()
from utils.structure import idx2coord, coord2idx


def translate_result(df):
    L = df.columns[df.values[0] == 1]
    table = np.zeros((len(L), 4))
    for m in range(len(L)):
        i, j = extract_idx(df.columns[df.values[0] == 1][m])
        x, y, z = idx2coord(j)
        table[m, 0] = i
        table[m, 1] = x
        table[m, 2] = y
        table[m, 3] = z
    return table


def table_to_map(table):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz
    
    data = np.zeros((len(args.basic.elements), nsites))
    for i in range(len(table)):
        idx = coord2idx(table[i, 1], table[i, 2], table[i, 3])
        data[int(table[i, 0]), idx] = 1
    return data


def extract_idx(s):
    pattern = r'x\[(\d+)\]\[(\d+)\]'
    match = re.search(pattern, s)
    
    if match:
        i = int(match.group(1))
        j = int(match.group(2))
        return i, j
    else:
        print("pattern not found")
        return 1, -1
    
    
def stack_dataframe(df, df_stack):
    if df_stack.empty:
        df_stack = df
    else:
        try:
            df_stack = pd.concat([df_stack, df], ignore_index=True)
        except Exception as e:
            print(e)
    return df_stack


def composition_grid_search(nsites, nspecies, grid_range, current=None):
    if not current:
        current = []
    if nspecies == 1:
        if grid_range[0][0] <= nsites <= grid_range[0][1]:
            yield current + [nsites]
        return
    else:
        na_min, na_max = grid_range[0]
        next_grid_range = grid_range[1:] if len(grid_range) > 1 else [(1, nsites)]
        for i in range(na_min, min(na_max, nsites - nspecies + 1) + 1):
            if nsites - i >= nspecies - 1:
                yield from composition_grid_search(nsites - i, nspecies - 1, next_grid_range, current + [i])
                
