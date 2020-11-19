import cmath
import math
import numpy as np
from numpy import linalg as la

d = 3   # qutrit

# qutrit X and Z matrices
omega = cmath.exp(cmath.pi*2j/d)
X = np.array([[0,0,1],[1,0,0],[0,1,0]])
Z = np.array([[1,0,0],[0,omega,0],[0,0,omega**2]])

# 1-qudit pauli matrices
T = np.array([[omega**(-2*a*b)*np.dot(la.matrix_power(Z,a),la.matrix_power(X,b)) for b in range(d)] for a in range(d)])

# 1-qudit phase space operators (TODO: change the sum below to a numpy sum)
A_0 = (1./d)*sum([sum([T[a,b] for b in range(d)]) for a in range(d)])
A = np.zeros((d**2,d,d), dtype=np.complex128)
for u in range(d**2):
    index = np.unravel_index(u,(d,d))
    A[u] = np.dot(T[index],np.dot(A_0,np.conj(T[index].T)))
ps_coefficients = 1./d*A.reshape((d**2,d**2))

# Computational basis C_u = |u_1><u_2|
#comp_basis = np.zeros((d,d,d,d), dtype=np.complex128)
#for i in range(d):
#    for j in range(d):
#        comp_basis[i,j,i,j] = 1
#comp_basis = comp_basis.reshape((d**2,d,d))

# Change of basis coefficients Tr[A_u^\dagger C_u] = Tr[A_u C_u]
#ps_coefficients = np.zeros((d**2,d**2),dtype=np.complex128)
#for i in range(d**2):
#    for j in range(d**2):
#        ps_coefficients[i,j] = 1./d*np.trace(np.dot(A[i],comp_basis[j]))

# change of basis coefficients
# TODO: change this to a numpy array
#ps_coefficients = [[np.trace(np.dot(A[u],comp_basis[v])) for u in range(d**2)] for v in range(d**2)]

#TODO: get rid of for loops
def basis_change(mpo, coefficients):
    n = len(mpo)
    
    res = [None]*n
    for k in range(n):
        res[k] = np.zeros(mpo[k].shape, dtype=np.complex128)
        temp = np.tensordot(mpo[k],coefficients,axes=([1,1])) # sum over physical indices
        temp = np.moveaxis(temp, [0,1,2], [0,2,1])
        res[k] = temp
    return res

# n-dimensional phase space operators:
def phase_space_ops(n, d):
    A_n = np.zeros((d**(2*n),d**n,d**n),dtype=np.complex128)
    tensor_shape = tuple([d**2]*n)
    for u in range(d**(2*n)):
        index = np.unravel_index(u,tensor_shape)
        temp = np.identity(1,dtype=np.complex128)
        for i in range(n):
            temp = np.kron(temp, A[index[i]])
        A_n[u] = temp
    return A_n

# Wigner function, magic estimates
# TODO: use MPO to compute wigner instead of state, so that you don't have to calculate A_n
def wigner_fct(state, n, d):
    A_n = phase_space_ops(n, d)
    
    W = np.zeros((d**(2*n)), dtype=np.complex128)
    M = 0. #old magic quantifier ("mana" in arxiv:1307.7171)
    N = 0. #new magic quantifier
    for u in range(d**(2*n)):
        W[u] = 1./(d**n)*np.trace(np.dot(state,A_n[u]))#TODO: should really be comparing ratio of .imag to .real
        if W[u].imag > 1e-14:
            print("Error! Wigner function must be real but W[%d] =" % u,repr(W[u]))
            return 
            
        if W[u].real < 0:
            M -= W[u].real
            N += (W[u].real)**2
            
    W = W.astype(np.float64)
    N = math.sqrt(N)
    return W, M, N

def basic_checks(d):
    # Check properties of phase space basis: Hermitian, Tr[A_uA_v] = d*delta_uv, Tr[A_u] = 1
    for i in range(d**2):
        if np.absolute(np.trace(A[i]) - 1.) > 1e-14:
            print(i, "Trace error: expected 1, got " + repr(np.trace(A[i])))
        for j in range(d**2):
            if abs(np.trace(np.dot(A[i],A[j])).imag) > 1e-14:
                print(i,j,"complex trace error!")
            if not(np.allclose(A[i].conj().T,A[i])):
                print(i,j,"hermiticity error!")
            if i == j:
                if abs(np.trace(np.dot(A[i],A[j])).real - d) > 1e-14:
                    print(i,j,"normalization error!")
            else:
                if abs(np.trace(np.dot(A[i],A[j])).real) > 1e-14:
                    print(i,j,"normalization error!")

# Check properties of n-qudit phase space basis: Hermitian, Tr[A_uA_v] = d^n*delta_uv, Tr[A_u] = 1
def verify_phase_space_properties(n, d):
    A_n = phase_space_ops(n, d)
    for i in range(d**(2*n)):
        if np.absolute(np.trace(A_n[i]) - 1.) > 1e-14:
            print(i, "Trace error: expected 1, got " + repr(np.trace(A_n[i])))
        for j in range(d**(2*n)):
            if abs(np.trace(np.dot(A_n[i],A_n[j])).imag) > 1e-14:
                print(i,j,"complex trace error!")
            if not(np.allclose(A_n[i].conj().T,A_n[i])):
                print(i,j,"hermiticity error!")
            if i == j:
                if abs(np.trace(np.dot(A_n[i],A_n[j])).real - (d**n)) > 1e-14:
                    print(i,j,"normalization error!")
            else:
                if abs(np.trace(np.dot(A_n[i],A_n[j])).real) > 1e-14:
                    print(i,j,"normalization error!")