from pyqubo import Array, Binary, Placeholder, Constraint
import numpy as np

from fm.utils import load_fm_model
from arguments import JSONParameters
args = JSONParameters()


class hamiltonian:
    def __init__(self, nspecies, nsites):
        self.nspecies = nspecies
        self.nsites = nsites
        self.x = Array.create('x', shape=(self.nspecies, self.nsites), vartype='BINARY')
        self.M = Placeholder('M')
        self.H = 0
        
    def construct_hamiltonian(self, model_txt):
        Q, offset = load_fm_model(model_txt, self.nspecies * self.nsites, self.nsites)
        for i in range(self.nspecies):
            for j in range(self.nsites):
                for k in range(self.nspecies):
                    for l in range(self.nsites):
                        self.H += Q[i * self.nsites + j, k * self.nsites + l] * self.x[i, j] * self.x[k, l]
                        
    def apply_constraints(self, composition, mode, scale):
        K1 = self.M if mode == 'tight' else self.M * scale
        for i in range(self.nspecies):
            self.H += K1 * (sum(self.x[i, :]) - composition[i]) ** 2
            
        K2 = self.M
        for j in range(self.nsites):
            for i in range(self.nspecies):
                self.H += K2 * self.x[i, j] * (sum(self.x[:, j]) - 1)
                
    def compile_hamiltonian(self):
        return self.H.compile()
    
    def translate(self, coeff, obj):
        model = self.compile_hamiltonian()
        if obj == 'bqm':
            bqm = model.to_bqm(feed_dict={'M': coeff})
            return bqm
        elif obj == 'qubo':
            qubo, offset = model.to_qubo(feed_dict={'M': coeff})
            return qubo, offset
        elif obj == 'ising':
            ising, offset = model.to_ising(feed_dict={'M': coeff})
            return ising, offset
        else:
            print("Invalid translated format!!!")
            
