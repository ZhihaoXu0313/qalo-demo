import neal
from dwave.system import LeapHybridSampler, DWaveSampler, FixedEmbeddingComposite, FixedEmbeddingComposite
import dimod
from dimod import BinaryQuadraticModel
from dwave_qbsolv import QBSolv 

import networkx as nx 
import minorminer


def simulate_annealing(bqm):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm)
    return sampleset


def qbsolv_annealing(Q, subQuboSize=45):
    G = nx.complete_graph(subQuboSize)
    system = DWaveSampler()
    embedding = minorminer.find_embedding(G.edge, system.edgelist)
    sampler = FixedEmbeddingComposite(system, embedding)
    sampleset = QBSolv.sample_qubo(Q, solver=sampler, solver_limit=subQuboSize, label='QUBO Optimization')
    return sampleset


def hybrid_quantum_annealing(bqm):
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm)
    return sampleset

