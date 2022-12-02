# File containing all the various loss functions I have written
import jax.numpy as jnp
from NRT_functions import helper_functions

### LOSSES FOR CIRCLE, 3 different measures of separation quality, 1 for positivity, and 1 for equivariance
### ALSO THE SAME LOSSES AS FOR TORUS, ALL JUST WORKS!

def sep_circ_EucChi(W, I, Chi):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # Measure the separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    L_Sep = 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, Chi))
    return L_Sep

def sep_circ_Euc(W, I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # Measure the separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    L_Sep = 1/jnp.power(N,2)*jnp.sum(Xi)
    return L_Sep

def sep_circ_Kern(W, I, sigma_sq):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(Xi)/jnp.power(N,2)
def sep_circ_KernChi(W, I, sigma_sq, Chi):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    L_Sep = 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, Chi))
    return L_Sep

# Positivity loss linear
def pos_circ(W, I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    V = jnp.matmul(W, I)
    [D, N] = V.shape

    # Ignore all the positive entries
    V_Neg = (V - jnp.abs(V))/2

    # Then measure the positivity
    L_pos = -jnp.sum(V_Neg)/(D*N)
    return L_pos

# Same as above but learnable return matrix
def equi_circ_smart_B(W, B, I_base, I_shift, G_I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    V_base = jnp.matmul(W,I_base)
    V_shift = jnp.matmul(W,I_shift)
    G = jnp.einsum('ij,kjp->kip', W, jnp.einsum('ikl,lp->ikp', G_I, B))
    V_tilde = jnp.einsum('kij,jl->ilk',G, V_base)
    V_tilde_reshape = jnp.reshape(V_tilde, [V_tilde.shape[0], V_tilde.shape[1]*V_tilde.shape[2]],order='F')
    L = jnp.sum(jnp.power(V_tilde_reshape - V_shift,2))/(V_tilde_reshape.shape[1])
    return L

### LOSSES FOR LINE 3 different separation, positivity, and a norm one

def pos_line(W, om, phi, N_shift):
    # Create the irrep basis
    I = helper_functions.init_irreps_1D(om, phi)

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)/(N_shift+1)
    g = g/norms[:,None]
    [D, N] = g.shape

    # measure positivity
    g_neg = (g - jnp.abs(g))/2
    return -jnp.sum(g_neg)/(D*N)

def sep_line_Euc(W, om, phi):
    # Create the irrep basis
    I = helper_functions.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure positivity
    Xi = -jnp.sum(jnp.power(g[:, None, :] - g[:, :, None], 2), axis=0)
    return jnp.sum(Xi)/jnp.power(N,1)

def sep_line_Kern(W, om, phi, sigma_sq):
    # Create the irrep basis
    I = helper_functions.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(Xi)/jnp.power(N,2)

def sep_line_EucChi(W, om, phi, chi):
    # Create the irrep basis
    I = helper_functions.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    return 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, chi))

def sep_line_KernChi(W, om, phi, sigma_sq, chi):
    # Create the irrep basis
    I = helper_functions.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(jnp.multiply(Xi, chi))/jnp.power(N,3/2)

def norm_line(W, om, phi_room, phi_other):
    # Create the irrep basis
    I_room = helper_functions.init_irreps_1D(om, phi_room)
    I_other = helper_functions.init_irreps_1D(om, phi_other)

    # Use the room to normalise the other representations
    g_room = jnp.matmul(W, I_room)
    g_other = jnp.matmul(W, I_other)
    norms = jnp.linalg.norm(g_room, axis = 1)
    g = g_other/norms[:,None]
    [D, N] = g_room.shape
    N_shift = int(phi_other.size/N)

    # Measure the resulting norms in each of the rooms and penlise deviations from 1
    norms = jnp.sum(jnp.reshape(jnp.power(g_other, 2), [D, N_shift, N]), axis = 2)
    return jnp.linalg.norm(norms - 1)/(D*N_shift)

### LOSSES FOR PLANE 3 different separation, positivity, and a norm one
# SAME AS FOR LINE, JUST USING 2D IRREP INIT!

def pos_plane(W, om, phi, N_shift):
    # Create the irrep basis
    I = helper_functions.init_irreps_2D(om, phi)

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)/(N_shift+1)
    g = g/norms[:,None]
    [D, N] = g.shape

    # measure positivity
    g_neg = (g - jnp.abs(g))/2
    return -jnp.sum(g_neg)/(D*N)

def sep_plane_Euc(W, om, phi):
    # Create the irrep basis
    I = helper_functions.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure positivity
    Xi = -jnp.sum(jnp.power(g[:, None, :] - g[:, :, None], 2), axis=0)
    return jnp.sum(Xi)/jnp.power(N,2)

def sep_plane_Kern(W, om, phi, sigma_sq):
    # Create the irrep basis
    I = helper_functions.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(Xi)/jnp.power(N,2)

def sep_plane_EucChi(W, om, phi, chi):
    # Create the irrep basis
    I = helper_functions.init_irreps_2D(om, phi)
    N = phi.shape[0]

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    return 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, chi))

def sep_plane_KernChi(W, om, phi, sigma_sq, chi):
    # Create the irrep basis
    I = helper_functions.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(jnp.multiply(Xi, chi))/jnp.power(N,2)

def sep_plane_KernChi_Module(W, grid_params, phi, sigma_sq, chi):
    # Create the frequencies
    M = W.shape[0]
    M_Q = int(M/2)
    om_1 = helper_functions.freq_module_plane_new(grid_params[0:4], M_Q)
    om_2 = helper_functions.freq_module_place_new(grid_params[4:], M_Q)
    om = jnp.vstack([om_1, om_2])

    # Create the irrep basis
    I = helper_functions.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis=1)
    g = g / norms[:, None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(jnp.multiply(Xi, chi))/jnp.power(N,2)


def norm_plane(W, om, phi_room, phi_other):
    # Create the irrep basis
    I_room = helper_functions.init_irreps_2D(om, phi_room)
    I_other = helper_functions.init_irreps_2D(om, phi_other)

    # Use the room to normalise the other representations
    g_room = jnp.matmul(W, I_room)
    g_other = jnp.matmul(W, I_other)
    norms = jnp.linalg.norm(g_room, axis = 1)
    g = g_other/norms[:,None]
    [D, N] = g_room.shape
    N_shift = int(phi_other.shape[0]/N)

    # Measure the resulting norms in each of the rooms and penlise deviations from 1
    norms = jnp.sum(jnp.reshape(jnp.power(g_other, 2), [D, N_shift, N]), axis = 2)
    return jnp.linalg.norm(norms - 1)/(D*N_shift)


