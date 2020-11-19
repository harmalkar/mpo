import numpy as np
from numpy import linalg as la

def state_to_tensor(state, n, d):
    tensor_shape = tuple([d]*(2*n))
    tensor = state.reshape(tensor_shape)
    return tensor
    
def tensor_to_local_tensor(tensor, n, d):
    local_axes = [int(i/2) if i%2 == 0 else int(n + (i/2)) for i in range(2*n)]
    original_axes = [i for i in range(2*n)]
    return np.moveaxis(tensor, original_axes, local_axes)

def tensor_to_mpo(tensor, n, d, verbose = False, **kwargs):
    # If not given a maximum bond dimension, set it to the maximum possible - d^(4n) (TODO)?
    #TODO: add a truncation magnitude cuttoff instead of just max_bd
    max_bd = kwargs.get('max_bd', d**(4*n))
    
    # Preparing for first SVD: Constructing environment tensor T for first site
    d2 = d**2
    T = tensor.reshape((d2,d2**(n-1)))
    if verbose:
        print(0, "Initial T:\t\t\t\t", T.shape)

    # First site
    mpo = [0]*n
    U, S, Vt = la.svd(T, full_matrices = False)
    U = U[:,:max_bd]
    S = S[:max_bd]
    Vt = Vt[:max_bd,:]
    mpo[0] = U.reshape((1,U.shape[0],U.shape[1]))
    T = np.dot(np.diag(S),Vt).reshape(S.size*d2,int(Vt.shape[1]/d2))
    if verbose:
        print(1, U.shape, Vt.shape, "\t->", mpo[0].shape, "\t", T.shape, "\t", S.size)

    # Interior sites
    for i in range(1,n-1):
            U, S, Vt = la.svd(T, full_matrices = False)
            U = U[:,:max_bd]
            S = S[:max_bd]
            Vt = Vt[:max_bd,:]
            mpo[i] = U.reshape((int(U.shape[0]/d2),d2,U.shape[1]))
            T = np.dot(np.diag(S),Vt).reshape((S.size*d2,int(Vt.shape[1]/d2)))
            if verbose:
                print(i+1, U.shape, Vt.shape, "\t->", mpo[i].shape, "\t", T.shape, "\t", S.size)

    # Last site
    mpo[n-1] = np.dot(np.diag(S),Vt).reshape((S.size, Vt.shape[0], 1))
    if verbose:
        print(n, "Final Matrix:\t\t  ", mpo[n-1].shape)

    return mpo

# Given a density matrix, return its matrix product operator representation
# Indices are stored in the following format: (bond,phys,bond)
# Usage: state_to_mpo(state, n, d, verbose) or state_to_mpo(state, n, d, verbose, max_bd). verbose defaults to False if not given.
def state_to_mpo(state, n, d, verbose=False, **kwargs):
    tensor = state_to_tensor(state, n, d)
    local_tensor = tensor_to_local_tensor(tensor, n, d)
    return tensor_to_mpo(local_tensor, n, d, verbose, **kwargs)

# Frobenius inner product tr[A^\dagger B] between two states A and B represented by mpos 
# norm: Frobenius normalization of local basis elements used in MPO representation (norm = Tr[A_u^\dagger A_v])
# Assumptions:
#       len(m1) = len(m2)                                       [same length]
#       m1[i].shape = m2[i].shape = (bond,phys,bond)            [same bond and phys dim at each site]
#       m1[0].shape = (1,_,_) and m1[-1].shape = (_,_,1)        [closed boundary conditions]
def inner_prod(m1, m2, norm=1):
        # Extract length, phys dim, and first site bond dim
        n = len(m1)

        #print(m1[0][0].shape, m2[0][0].shape)
        # First site
        M = np.tensordot(m1[0][0].conj(),m2[0][0],axes=([0,0]))      # sum over physical indices (grab the first and only matrix due to fixed bdry cond.)

        #print(0,m1[0].shape,M.shape)
        # Rest of contraction
        for i in range(1,n):
                M = np.tensordot(M,m1[i].conj(),axes=([0,0]))
                M = np.tensordot(M,m2[i],axes=([0,1],[0,1]))
                #print(i,m1[i].shape, M.shape)
        return M[0][0]*norm
    
# Check that an MPO represents a state
def check_coefficients(mpo, state, n, d):
    # Calculate tensor of MPO coefficients
    mpo_tensor_shape = tuple([d**2]*n)
    mpo_tensor = np.zeros(mpo_tensor_shape, dtype=np.complex128)
    for u in range(d**(2*n)):
        # Calculate coefficient
        index = np.unravel_index(u,mpo_tensor_shape)
        coefficient = np.identity(1,dtype=np.complex128)
        for i in range(n):
            coefficient = np.dot(coefficient, mpo[i][:,index[i],:])
        mpo_tensor[index] = coefficient[0][0]
        
    # Reshape into state
    state_tensor_shape = tuple([d]*(2*n))
    mpo_state = mpo_tensor.reshape(state_tensor_shape)
    tensor_axes = [i for i in range(2*n)]
    T_axes = [int(i/2) if i%2 == 0 else int(n + (i/2)) for i in range(2*n)]
    mpo_state = np.moveaxis(mpo_state, T_axes, tensor_axes)
    mpo_state = mpo_state.reshape((d**n,d**n))
    print(np.allclose(mpo_state, state))
    
