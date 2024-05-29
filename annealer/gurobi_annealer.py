import gurobipy as gp 
from gurobipy import GRB 
import numpy as np 
from arguments import JSONParameters
args = JSONParameters()


def gurobi_annealing(Q, time_limit=None, gap_limit=None):
    Nx, Ny, Nz = args.structure.sc[0], args.structure.sc[1], args.structure.sc[2]
    nsites = int(len(args.structure.site)) * Nx * Ny * Nz
    nspecies = len(args.basic.elements)
    
    quboMat = np.zeros((nsites * nspecies, nsites * nspecies))
    for i in range(nspecies):
        for j in range(nsites):
            for k in range(nspecies):
                for l in range(nsites):
                    try:
                        quboMat[i * nsites + j, k * nsites + l] = Q[('x[' + str(i) + '][' + str(j) + ']', 'x[' + str(k) + '][' + str(l) + ']')]
                    except KeyError:
                        pass
    quboMat = list(quboMat)
    model = gp.Model("QUBO")
    n = len(quboMat)
    
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if gap_limit is not None:
        model.setParam(GRB.Param.MIPGap, gap_limit)
        
    variables = [model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(n)]
    objective = sum(quboMat[i][j] * variables[i] * variables[j] for i in range(n) for j in range(n))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    solution = [int(v.x) for v in variables]
    return solution
        
