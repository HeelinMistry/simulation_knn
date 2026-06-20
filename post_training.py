import sys; sys.path.insert(0, '.')
from agents.mc_knn_memory import MCKNNMemory
mem = MCKNNMemory.load('outcomes/mc_knn_agent_best.npz')
print('bank size:', len(mem))
print('n_commits:', mem.n_commits, 'n_prunes:', mem.n_prunes)
import numpy as np
print('return stats: mean', mem.returns[:len(mem)].mean(), 'std', mem.returns[:len(mem)].std())
print('action balance:', np.bincount(mem.actions[:len(mem)], minlength=4))
