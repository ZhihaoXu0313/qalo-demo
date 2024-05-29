import os
import numpy as np 
import pandas as pd 

from annealer.qubo import hamiltonian
from annealer.utils import *
from annealer.gurobi_annealer import gurobi_annealing
from annealer.quantum_annealer import *
from arguments import JSONParameters
args = JSONParameters()


class AnnealerEngine:
    def __init__(self, nspecies, nsites, placeholder, fmpath, composition, annealer, mode, ks):
        self.nspecies = nspecies
        self.nsites = nsites
        self.placeholder = placeholder
        
        self.hQubo = hamiltonian(nspecies, nsites)
        self.hQubo.construct_hamiltonian(os.path.join(fmpath, "model.txt"))
        self.hQubo.apply_constraints(composition, mode, ks)
        self.hQubo.compile_hamiltonian()
        
        self.bqm = self.hQubo.translate(placeholder, 'bqm')
        self.Q, self.offset = self.hQubo.translate(placeholder, 'qubo')
        
        self.annealer = annealer
        
        self.structuredfstack = pd.DataFrame()


    def run_annealer(self, n_sim):
        if self.annealer == 'gurobi':
            for _ in range(n_sim):
                result = gurobi_annealing(self.Q, time_limit=120, gap_limit=0.0001)
                self.structure2dmap += np.array(result).reshape(-1, self.nsites) / n_sim
                dfs = pd.DataFrame(result)
                dfs = dfs.T 
                dfs.columns = [f'x_{col}' for col in dfs.columns]
                self.structuredfstack = stack_dataframe(dfs, self.structuredfstack)
        elif self.annealer == 'qasim':
            for _ in range(n_sim):
                sampleset = simulate_annealing(self.bqm)
                dfs = pd.DataFrame(sampleset.lowest())
                self.structuredfstack = stack_dataframe(dfs, self.structuredfstack)
                result = translate_result(dfs)
        elif self.annealer == 'hybrid':
            for _ in range(n_sim):
                sampleset = hybrid_quantum_annealing(self.bqm)
                dfs = pd.DataFrame(sampleset.lowest())
                self.structuredfstack = stack_dataframe(dfs, self.structuredfstack)
                result = translate_result(dfs)
    
    def extract_solutions(self):
        columns_sorted = sorted(self.structuredfstack.columns, key=extract_idx)
        return self.structuredfstack[columns_sorted]
    


    
    