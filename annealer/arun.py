import numpy as np 
import os
import time
import pandas as pd 

from annealer.engine import AnnealerEngine
from annealer.utils import composition_grid_search
from mlp.snap import snap_model_inference
from utils.data import add_new_structure, save_energy
from utils.utils import plot2dstruct

from arguments import JSONParameters
args = JSONParameters()


def annealing_GridSearch(composition_range, annealer):
    num = 0
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz
    nspecies = len(args.basic.elements)
    
    t_start = time.time()
    for i, composition in enumerate(composition_grid_search(nsites, nspecies, composition_range), start=1):
        annealing_obj = AnnealerEngine(nspecies=nspecies, nsites=nsites, 
                                       placeholder=args.quantum_annealer.constr, fmpath=args.work_path.fm,
                                       composition=composition, annealer=annealer, mode='tight', 
                                       ks=1)
        annealing_obj.run_annealer(n_sim=args.quantum_annealer.shots)
        structuredfstack = annealing_obj.extract_solutions()
        energy = []
        for n, s in structuredfstack.iterrows():
            e = snap_model_inference(binvec=np.array(s.values), 
                                     infile=os.path.join(args.work_path.lmps, args.mlp.lmp_infile), 
                                     coeffile=os.path.join(args.work_path.lmps, args.mlp.lmp_coeffile), 
                                     path_of_tmp=args.work_path.tmp)
            energy.append(e)
        add_new_structure(energy=energy, dfstack=structuredfstack, 
                          fmpath=args.work_path.fm, newpath=args.work_path.new_data)
        csvfile = "full-energy-" + str(i) + ".csv"
        save_energy(os.path.join(args.work_path.output, csvfile), energy, composition)
        num = i
    t_end = time.time()
    print("grid search mode finished!")
    print("total composition optimized: " + str(num))
    print("total annealer wall time: " + str(t_end - t_start))
    
    
def annealing_AutoSearch(initial_composition, mix_circle, annealer):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz
    nspecies = len(args.basic.elements)
    
    t_start = time.time()
    composition = initial_composition
    energy_solution = []
    structuredf = pd.DataFrame()
    
    initial_k = args.quantum_annealer.relax * args.quantum_annealer.constr
    growth_rate = args.quantum_annealer.relax
    
    for i in range(mix_circle):
        k = growth_rate ** (1 - i / mix_circle)
        annealing_loose = AnnealerEngine(nspecies=nspecies, nsites=nsites,
                                         placeholder=args.quantum_annealer.constr, fmpath=args.work_path.fm,
                                         composition=composition, annealer=annealer, mode="loose", 
                                         ks=k)
        annealing_loose.run_annealer(n_sim=args.quantum_annealer.shots)
        sdfl = annealing_loose.extract_solutions()
        energy = []
        energy_min = 0
        for n, s in sdfl.iterrows():
            e = snap_model_inference(binvec=np.array(s.values), 
                                     infile=os.path.join(args.work_path.lmps, args.mlp.lmp_infile), 
                                     coeffile=os.path.join(args.work_path.lmps, args.mlp.lmp_coeffile), 
                                     path_of_tmp=args.work_path.tmp)
            energy.append(e)
            energy_min = min(energy_min, e)
        
        if i != mix_circle - 1:
            min_index = energy.index(energy_min)
            min_structure = sdfl.iloc[min_index, :].tolist()
            for j in range(nspecies):
                composition[j] = sum(min_structure[j * nsites: (j + 1) * nsites])
        else:
            energy_solution = energy
            structuredf = sdfl
            
    add_new_structure(energy=energy_solution, dfstack=structuredf, 
                      fmpath=args.work_path.fm, newpath=args.work_path.new_data)
    csvfile = "mix-energy-composition.csv"
    save_energy(os.path.join(args.work_path.output, csvfile), energy_solution, composition)
    
    t_end = time.time()
    print("auto search mode finished!")
    print("start compostion: ", initial_composition)
    print("end composition: ", composition)
    print("total annealer wall time: " + str(t_end - t_start))
    return composition
        
