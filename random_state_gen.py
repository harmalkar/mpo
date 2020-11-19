import numpy as np
from numpy import linalg as la

# Generate a mixed state of randomly sampled pure qudit states                                                 
# d - onsite hilbert space dimension
# N - number of sites
# M - number of pure states used
def random_mixed_state(N,d=3,M=10):
        states = [random_state(N,d) for i in range(M)]
        weights = [np.random.uniform(low=0.0,high=1.0) for i in range(M)]
        total_weight = sum(weights)
        state = np.zeros((d**N,d**N))
        for i in range(M):
                state = state + (weights[i]/total_weight) * np.outer(states[i],states[i].conj())
        return state

# Generate a random qudit state
# d - onsite hilbert space dimension
# N - number of sites
def random_state(N,d=3):
        normals = [np.random.standard_normal(size=d**N) for i in range(2)]
        state = np.array([normals[0][i] + 1j * normals[1][i] for i in range(d**N)])
        state /= la.norm(state)
        return state