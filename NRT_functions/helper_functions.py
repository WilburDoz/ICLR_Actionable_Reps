# Helper functions for neural representation theory
import os
from datetime import datetime 
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from scipy.special import sph_harm as sph_harm
import pickle

# A function to setup a save file
def setup_save_file(parameters):
    today = datetime.strftime(datetime.now(), '%y%m%d')
    now = datetime.strftime(datetime.now(), '%H%M%S')
    # Make sure folder is there
    if not os.path.isdir(f"./data/"):
        os.mkdir(f"./data/")
    if not os.path.isdir(f"data/{today}/"):
        os.mkdir(f"data/{today}/")
    # Now make a folder in there for this run
    savepath = f"data/{today}/{now}/"
    if not os.path.isdir(f"data/{today}/{now}"):
        os.mkdir(f"data/{today}/{now}")

    save_obj(parameters, "parameters", savepath)
    return savepath

# A function to implement an ADAM step
def adam_step(grad, m, v, t, epsilon, beta1, beta2, eta):
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*jnp.power(grad, 2)
    m_hat = m/(1-jnp.power(beta1, t))
    v_hat = v/(1-jnp.power(beta2, t))
    return epsilon*m_hat/(jnp.sqrt(v_hat) + eta), m, v

def save_obj(obj, name, savepath):
    with open(savepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, filepath):
    with open(filepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Function to save weights
def save_weights(W, counter):
    today =  datetime.strftime(datetime.now(),'%y%m%d')
    now = datetime.strftime(datetime.now(),'%H%M%S')

    # Make sure folder is there
    if not os.path.isdir(f"figures/{today}/weights/"):
        os.mkdir(f"figures/{today}/weights/")

    csv_file = f"./figures/{today}/weights/{now}_{counter}.csv"

    # Note the saving if this is the first weight file
    if counter == 0:
        text_file = f"./figures/{today}/plot_log.txt"
        Path(text_file).touch()
        file = open(text_file,"a")
        file.write(f"At {now} a set of weights was saved\n")
        file.close()

    np.savetxt(csv_file, W)

# Initiaised a set of irreps at the given frequenies and angles
def init_irreps_1D(om, phi):
    cos_stack = jnp.cos(phi[None,:]*om[:,None])
    sin_stack = jnp.sin(phi[None,:]*om[:,None])
    I = jnp.ones([2*om.size + 1, phi.size])
    I = I.at[1::2,:].set(cos_stack)
    I = I.at[2::2,:].set(sin_stack)
    return I

def init_irreps_2D(om, phi):
    cos_stack = jnp.cos(jnp.sum(phi[None,:,:]*om[:,None,:], axis = 2))
    sin_stack = jnp.sin(jnp.sum(phi[None,:,:]*om[:,None,:], axis = 2))
    I = jnp.ones([2*jnp.shape(om)[0] + 1, jnp.shape(phi)[0]])
    I = I.at[1::2,:].set(cos_stack)
    I = I.at[2::2,:].set(sin_stack)
    return I

def init_irreps_AB(om, theta_A, theta_B, phi):
    cos_stack_A = jnp.cos(theta_A[None, :]*om[:, None])
    sin_stack_A = jnp.sin(theta_A[None, :]*om[:, None])
    cos_stack_B = jnp.cos(theta_B[None, :]*om[:, None])
    sin_stack_B = jnp.sin(theta_B[None, :]*om[:, None])
    I = jnp.ones([8*om.size + 1, theta_A.size])
    I = I.at[1::8, :].set(cos_stack_A)
    I = I.at[2::8, :].set(sin_stack_A)
    I = I.at[3::8, :].set(cos_stack_B)
    I = I.at[4::8, :].set(sin_stack_B)
    I = I.at[5::8, :].set(cos_stack_A*jnp.power(-1, phi)[None, :])
    I = I.at[6::8, :].set(cos_stack_A*jnp.power(-1, phi)[None, :])
    I = I.at[7::8, :].set(sin_stack_A*jnp.power(-1, phi)[None, :])
    I = I.at[8::8, :].set(sin_stack_A*jnp.power(-1, phi)[None, :])
    return I

# Finds just the set of transformations matrices for the list of delta_phi angles
def irrep_transforms_1D(om, delta_phi):
    N = delta_phi.size
    M = om.size
    # Here we're going to create the full set of irrep transformation matrices
    G_I = np.zeros([N,2*M+1,2*M+1])
    G_I[:,0, 0] = 1
    counter = 0
    for delta in delta_phi:
        for m in range(1,M+1):
            G_I[counter, 2*m-1:2*m+1,2*m-1:2*m+1] = [[np.cos(om[m-1]*delta), -np.sin(om[m-1]*delta)],[np.sin(om[m-1]*delta), np.cos(om[m-1]*delta)]]
        counter = counter + 1
    return G_I

def irrep_transforms_2D(om, delta_phis):
    N = delta_phis.shape[0]
    M = om.shape[0]
    G_I = np.zeros([N, 2*M+1, 2*M+1])

    G_I[:,0,0] = 1

    for counter in range(N):
        for m in range(1, M+1):
            delta = om[m-1,0]*delta_phis[counter,0] + om[m-1,1]*delta_phis[counter,1]
            G_I[counter, 2*m-1:2*m+1, 2*m-1:2*m+1] = [[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]]
    return G_I

# Function to normalise the weights along the rows
def normalise_weights(W):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]
    return W

def calc_chi_circ(phi, sigma_theta, f):
    phi = jnp.asarray(phi)
    circ_dist_sq = jnp.amin(jnp.asarray([jnp.power(phi[:,None] - phi[None,:],2),jnp.power(phi[:,None] - phi[None,:]-2*jnp.pi,2),jnp.power(phi[:,None] - phi[None,:]+2*jnp.pi,2)]),0)
    Chi = 1 - f*jnp.exp(-circ_dist_sq/(2*jnp.power(sigma_theta,2)))
    return Chi

def calc_chi_line(phi, sigma_theta, f):
    dist = jnp.power(phi[:,None] - phi[None,:],2)
    Chi = 1 - f*jnp.exp(-dist/(2*jnp.power(sigma_theta,2)))
    return Chi

def calc_chi_line_euc(phi, sigma_theta, f):
    Chi = jnp.power(phi[:,None] - phi[None,:],2)
    return Chi

def calc_chi_torus(phi, sigma_theta, f):
    phi = jnp.asarray(phi)
    circ_dist_sq_th = jnp.amin(jnp.asarray([jnp.power(phi[None,:,0] - phi[:,None,0],2),jnp.power(phi[None,:,0] - phi[:,None,0]-2*jnp.pi,2),jnp.power(phi[None,:,0] - phi[:,None,0]+2*jnp.pi,2)]),0)
    circ_dist_sq_ph = jnp.amin(jnp.asarray([jnp.power(phi[:,None,1] - phi[None,:,1],2),jnp.power(phi[:,None,1] - phi[None,:,1]-2*jnp.pi,2),jnp.power(phi[:,None,1] - phi[None,:,1]+2*jnp.pi,2)]),0)
    chi = 1 - f*jnp.multiply(jnp.exp(-circ_dist_sq_th/(2*jnp.power(sigma_theta, 2))), jnp.exp(-circ_dist_sq_ph/(2*jnp.power(sigma_theta, 2))))
    return chi

def calc_chi_periodicvolume(phi, sigma_theta, f):
    phi = jnp.asarray(phi)
    circ_dist_sq_th = jnp.amin(jnp.asarray([jnp.power(phi[:,None,0] - phi[None,:,0],2),jnp.power(phi[:,None,0] - phi[None,:,0]-2*jnp.pi,2),jnp.power(phi[:,None,0] - phi[None,:,0]+2*jnp.pi,2)]),0)
    circ_dist_sq_ph = jnp.amin(jnp.asarray([jnp.power(phi[:,None,1] - phi[None,:,1],2),jnp.power(phi[:,None,1] - phi[None,:,1]-2*jnp.pi,2),jnp.power(phi[:,None,1] - phi[None,:,1]+2*jnp.pi,2)]),0)
    circ_dist_sq_ps = jnp.amin(jnp.asarray([jnp.power(phi[:,None,2] - phi[None,:,2],2),jnp.power(phi[:,None,2] - phi[None,:,2]-2*jnp.pi,2),jnp.power(phi[:,None,2] - phi[None,:,2]+2*jnp.pi,2)]),0)
    chi = 1 - f*jnp.multiply(jnp.exp(-circ_dist_sq_th/(2*jnp.power(sigma_theta, 2))), jnp.multiply(jnp.exp(-circ_dist_sq_ph/(2*jnp.power(sigma_theta, 2))), jnp.exp(-circ_dist_sq_ps/(2*jnp.power(sigma_theta, 2)))))
    return chi

def calc_chi_plane(phi, sigma_theta, f):
    dist = jnp.sum(jnp.power(phi[:,None,:] - phi[None,:,:],2), axis = 2)
    Chi = 1 - f*jnp.exp(-dist/(2*jnp.power(sigma_theta,2)))
    return Chi

def calc_chi_plane_euc(phi, sigma_theta, f):
    Chi = jnp.sum(jnp.power(phi[:,None,:] - phi[None,:,:],2), axis = 2)
    return Chi

def calc_chi_plane_exp(phi, sigma_theta, f):
    dist = jnp.sum(jnp.power(phi[:,None,:] - phi[None,:,:],2), axis = 2)
    Chi = jnp.exp(dist/(2*jnp.power(sigma_theta,2)))
    return Chi

def calc_chi_sphere(phi, sigma_theta):
    phi = jnp.asarray(phi)
    N_rand = phi.shape[1]
    delta_lat = phi[1,:,None] - phi[1,None,:]
    distance = jnp.arccos(jnp.sin(phi[0,:,None])*jnp.sin(phi[0,None,:]) + jnp.multiply(jnp.cos(phi[0,:,None])*jnp.cos(phi[0,None,:]), jnp.cos(delta_lat)))
    for ind in range(N_rand):
        distance = distance.at[ind,ind].set(0)
    chi = 1 - jnp.exp(-jnp.power(distance, 2)/(2*jnp.power(sigma_theta,2)))
    return chi

# This one is different, selects frequencies from half the plane
def freq_selector(M):
    # This section will pull out all the best frequencies from the semicircle
    M_Here = M + int(np.sqrt(M) - 1) + 1
    buffer = 10
    M_Lim = int(np.ceil(np.sqrt(M))) + buffer  # to take account of 0 mode
    indices_long = np.arange(-M_Lim, M_Lim + 1)
    indices_long_sq = np.power(indices_long, 2)
    indices_short = np.arange(M_Lim + 1)
    indices_short_sq = np.power(indices_short, 2)
    dist = indices_long_sq[None, :] + indices_short_sq[:, None]
    sorting_order = np.argsort(np.ndarray.flatten(dist))

    ind = np.zeros([M_Lim + 1, 2 * M_Lim + 1,2])
    ind[:, :, 0] = indices_long[None, :]
    ind[:, :, 1] = indices_short[:, None]

    chosen_indices = sorting_order[0:M_Here]
    ind_x = np.reshape(ind[:, :, 0], [(2 * M_Lim + 1) * (M_Lim + 1)])[chosen_indices]
    ind_y = np.reshape(ind[:, :, 1], [(2 * M_Lim + 1) * (M_Lim + 1)])[chosen_indices]
    inds =  np.transpose(np.stack([ind_x, ind_y], axis = 0))

    # Now cut out all the ones on the axis that are just negatives of one another
    inds = inds[np.logical_not(np.logical_and(inds[:,0] < 0, inds[:,1] == 0)), :]
    inds = inds[0:M, :]
    return inds

def freq_selector_3D(M):
    # This section will pull out all the best frequencies from the semicircle
    M_Here = M + int(np.power(M, 1 / 3) + 1) + 100
    buffer = 100
    M_Lim = int(np.ceil(np.sqrt(M))) + buffer  # to take account of 0 mode
    indices_long = np.arange(-M_Lim, M_Lim + 1)
    indices_long_sq = np.power(indices_long, 2)
    indices_short = np.arange(M_Lim + 1)
    indices_short_sq = np.power(indices_short, 2)
    dist = indices_long_sq[None, None, :] + indices_long_sq[None, :, None] + indices_short_sq[:, None, None]
    sorting_order = np.argsort(np.ndarray.flatten(dist))

    ind = np.zeros([M_Lim + 1, 2 * M_Lim + 1, 2*M_Lim + 1, 3])
    ind[:, :, :, 0] = indices_long[None, None, :]
    ind[:, :, :, 1] = indices_long[None, :, None]
    ind[:, :, :, 2] = indices_short[:, None, None]

    chosen_indices = sorting_order[0:M_Here]
    ind_x = np.reshape(ind[:, :, :, 0], [(2 * M_Lim + 1) * (M_Lim + 1) * (2*M_Lim + 1)])[chosen_indices]
    ind_y = np.reshape(ind[:, :, :, 1], [(2 * M_Lim + 1) * (M_Lim + 1) * (2*M_Lim + 1)])[chosen_indices]
    ind_z = np.reshape(ind[:, :, :, 2], [(2 * M_Lim + 1) * (M_Lim + 1) * (2*M_Lim + 1)])[chosen_indices]
    inds =  np.transpose(np.stack([ind_x, ind_y, ind_z], axis = 0))

    # Now cut out all the ones on the axis that are just negatives of one another
    inds = inds[np.logical_not(np.logical_and(inds[:,0] < 0, inds[:,1] == 0)), :]
    inds = inds[0:M, :]
    return inds

def freqs_grid_torus(base_lengthscale, relative_scale, relative_angle, M):
    buffer = 20
    M_Lim = int(np.ceil(np.sqrt(M))) + buffer  # to take account of 0 mode
    second_base_vector = np.rint(base_lengthscale * relative_scale * np.array([np.cos(relative_angle), np.sin(relative_angle)]))

    # Generate the first axis of freqs
    indices_short = np.hstack([np.arange(-M_Lim, M_Lim + 1)[:, None] * base_lengthscale, np.zeros(2 * M_Lim + 1)[:, None]])
    indices_long = np.arange(-M_Lim, M_Lim + 1)
    inds = np.reshape(np.moveaxis(indices_short[:, :, None] + second_base_vector[None, :, None] * indices_long[None, None, :], 1, 2), [np.size(indices_long) * np.shape(indices_short)[0], 2])
    inds = inds[inds[:, 0] > -0.00001, :]  # Keep only one half
    inds = inds[np.logical_not(np.logical_and(inds[:, 1] < 0, inds[:, 0] < 0.000001)),:]  # and chop out repeats on the y axis
    distance = np.power(inds[:, 0], 2) + np.power(inds[:, 1], 2)
    sorting_order = np.argsort(distance)
    inds = inds[sorting_order[0:M], :]

    return inds

def freqs_grid_plane(base_lengthscale, relative_scale, relative_angle, M, rotation_angle=0):
    buffer = 20
    M_Lim = int(np.ceil(np.sqrt(M))) + buffer  # to take account of 0 mode
    second_base_vector = base_lengthscale * relative_scale * np.array([np.cos(relative_angle), np.sin(relative_angle)])

    # Generate the first axis of freqs
    indices_short = np.hstack([np.arange(-M_Lim, M_Lim + 1)[:, None] * base_lengthscale, np.zeros(2 * M_Lim + 1)[:, None]])
    indices_long = np.arange(-M_Lim, M_Lim + 1)
    inds = np.reshape(np.moveaxis(indices_short[:, :, None] + second_base_vector[None, :, None] * indices_long[None, None, :], 1, 2), [np.size(indices_long) * np.shape(indices_short)[0], 2])
    inds = inds[inds[:, 0] > -0.00001, :]  # Keep only one half
    inds = inds[np.logical_not(np.logical_and(inds[:, 1] < 0, inds[:, 0] < 0.000001)),:]  # and chop out repeats on the y axis
    inds = inds[np.logical_not(np.logical_and(np.abs(inds[:,0]) < 0.001, np.abs(inds[:,1]<0.1)))] # remove constant freq
    distance = np.power(inds[:, 0], 2) + np.power(inds[:, 1], 2)
    sorting_order = np.argsort(distance)
    inds = inds[sorting_order[0:M], :]

    inds_rot = np.copy(inds)
    if rotation_angle == 0:
        inds_rot = inds
    else:
        # For some completely weird reason doing matrix multiplication broke...
        # So, since it only happens once per optimisation I did it manually...
        for freq in range(M):
            inds_rot[freq, 0] = np.cos(rotation_angle)*inds[freq, 0] - np.sin(rotation_angle)*inds[freq, 1]
            inds_rot[freq, 1] = np.cos(rotation_angle)*inds[freq, 1] + np.sin(rotation_angle)*inds[freq, 0]
    return inds_rot

def freq_module_plane_new(grid_params, M):
    buffer = 20  # to take account of 0 mode

    # Generate the first axis of freqs
    indices_short = jnp.arange(-buffer, buffer + 1)[:, None] * grid_params[0:2]
    indices_long = jnp.arange(-buffer, buffer + 1)
    inds = jnp.reshape(jnp.moveaxis(indices_short[:, :, None] + grid_params[2:4][None, :, None] * indices_long[None, None, :], 1, 2),
        [jnp.size(indices_long) * jnp.shape(indices_short)[0], 2])
    inds = inds[inds[:, 0] > -0.00001, :]  # Keep only one half
    inds = inds[jnp.logical_not(jnp.logical_and(inds[:, 1] < 0, inds[:, 0] < 0.000001)),
           :]  # and chop out repeats on the y axis
    inds = inds[jnp.logical_not(
        jnp.logical_and(jnp.abs(inds[:, 0]) < 0.001, jnp.abs(inds[:, 1] < 0.1)))]  # remove constant freq
    distance = jnp.power(inds[:, 0], 2) + jnp.power(inds[:, 1], 2)
    sorting_order = jnp.argsort(distance)
    inds = inds[sorting_order[0:M], :]

    return inds

def normalising_matrix(om):
    M = om.size

    c = jnp.sin(2*jnp.pi*om)/om
    s = (1 - jnp.cos(2*jnp.pi*om))/om
    sc = 0.5*((1 - jnp.cos(2*jnp.pi*(om[:,None] + om[None, :])))/(om[:,None] + om[None, :]) - (1 - jnp.cos(2*jnp.pi*(om[:,None] - om[None, :])))/(om[:,None] - om[None, :]))
    sc = sc.at[jnp.diag_indices(M)].set((1 - jnp.cos(4*jnp.pi*om))/(4*om))
    c2 = 0.5*(jnp.sin(2*jnp.pi*(om[:,None]+om[None,:]))/(om[:,None]+om[None,:]) + jnp.sin(2*jnp.pi*(om[:,None]-om[None,:]))/(om[:,None]-om[None,:]))
    c2 = c2.at[jnp.diag_indices(M)].set(0.5*(jnp.pi + jnp.sin(4*jnp.pi*om)/(2*om)))
    s2 = 0.5*(jnp.sin(2*jnp.pi*(om[:,None]-om[None,:]))/(om[:,None]-om[None,:]) - jnp.sin(2*jnp.pi*(om[:,None]+om[None,:]))/(om[:,None]+om[None,:]))
    s2 = s2.at[jnp.diag_indices(M)].set(0.5*(jnp.pi - jnp.sin(4*jnp.pi*om)/(2*om)))


    Q = 2*jnp.pi*jnp.ones([2*M+1,2*M+1])
    Q = Q.at[0,1:M+1].set(c)
    Q = Q.at[0,M+1:].set(s)
    Q = Q.at[1:M+1,0].set(c)
    Q = Q.at[M+1:,0].set(s)
    Q = Q.at[1:M+1,M+1:].set(sc)
    Q = Q.at[M+1:,1:M+1].set(sc)
    Q = Q.at[1:M+1,1:M+1].set(c2)
    Q = Q.at[M+1:,M+1:].set(s2)
    return Q

def normalising_matrix(om):
    M = om.size

    c = jnp.sin(2*jnp.pi*om)/om
    s = (1 - jnp.cos(2*jnp.pi*om))/om
    sc = 0.5*((1 - jnp.cos(2*jnp.pi*(om[:,None] + om[None, :])))/(om[:,None] + om[None, :]) - (1 - jnp.cos(2*jnp.pi*(om[:,None] - om[None, :])))/(om[:,None] - om[None, :]))
    sc = sc.at[jnp.diag_indices(M)].set((1 - jnp.cos(4*jnp.pi*om))/(4*om))
    c2 = 0.5*(jnp.sin(2*jnp.pi*(om[:,None]+om[None,:]))/(om[:,None]+om[None,:]) + jnp.sin(2*jnp.pi*(om[:,None]-om[None,:]))/(om[:,None]-om[None,:]))
    c2 = c2.at[jnp.diag_indices(M)].set(0.5*(jnp.pi + jnp.sin(4*jnp.pi*om)/(2*om)))
    s2 = 0.5*(jnp.sin(2*jnp.pi*(om[:,None]-om[None,:]))/(om[:,None]-om[None,:]) - jnp.sin(2*jnp.pi*(om[:,None]+om[None,:]))/(om[:,None]+om[None,:]))
    s2 = s2.at[jnp.diag_indices(M)].set(0.5*(jnp.pi - jnp.sin(4*jnp.pi*om)/(2*om)))


    Q = 2*jnp.pi*jnp.ones([2*M+1,2*M+1])
    Q = Q.at[0,1:M+1].set(c)
    Q = Q.at[0,M+1:].set(s)
    Q = Q.at[1:M+1,0].set(c)
    Q = Q.at[M+1:,0].set(s)
    Q = Q.at[1:M+1,M+1:].set(sc)
    Q = Q.at[M+1:,1:M+1].set(sc)
    Q = Q.at[1:M+1,1:M+1].set(c2)
    Q = Q.at[M+1:,M+1:].set(s2)
    return Q

def convert_angles(xyz):
    ptsnew = np.zeros([2, np.size(xyz, 1)])
    xy = xyz[0,:]**2 + xyz[1,:]**2
    #ptsnew[0,:] = np.arctan2(np.sqrt(xy), xyz[2,:]) # for elevation angle defined from Z-axis down
    ptsnew[0,:] = np.arctan2(xyz[2,:], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[1,:] = np.arctan2(xyz[1,:], xyz[0,:])
    return ptsnew

def initialise_irreps_sphere(ell_max, phi):
    N = phi.shape[1]
    I_C = Create_CSHs(ell_max, phi)
    I = Complex_to_Real_SHs(I_C, ell_max)
    return I

def Complex_to_Real_SHs(J_Comp, ell_max):
    J_Real = jnp.zeros(J_Comp.shape)
    l_ind = 0
    for l in range(ell_max+1):
        for m in range(-l, l+1):
            if m < 0:
                J_Real = J_Real.at[l_ind+l+m,:].set(1j/np.sqrt(2)*(J_Comp[l_ind+l+m,:] - (-1)**m*J_Comp[l_ind+l-m,:]))
            elif m == 0:
                J_Real = J_Real.at[l_ind+l,:].set(J_Comp[l_ind+l,:])
            else:
                J_Real = J_Real.at[l_ind+l+m,:].set(1/np.sqrt(2)*(J_Comp[l_ind+l-m,:] + (-1)**m*J_Comp[l_ind+l+m,:]))
        l_ind = l_ind + 2*l + 1
    return J_Real
def Real_to_Complex_SHs(J_Real, ell_max):
    J_Comp = jnp.zeros(J_Real.shape, dtype=complex)
    l_ind = 0
    for l in range(ell_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                J_Comp = J_Comp.at[l_ind + l + m, :].set(
                    1 / np.sqrt(2) * (J_Real[l_ind + l - m, :] - 1j * J_Real[l_ind + l + m, :]))
            elif m == 0:
                J_Comp = J_Comp.at[l_ind + l, :].set(J_Real[l_ind + l, :])
            else:
                J_Comp = J_Comp.at[l_ind + l + m, :].set(
                    (-1) ** m / np.sqrt(2) * (J_Real[l_ind + l + m, :] + 1j * J_Real[l_ind + l - m, :]))
        l_ind = l_ind + 2 * l + 1
    return J_Comp


def Create_CSHs(ell_max, angles):
    number_of_modes = np.power(ell_max + 1, 2)
    number_of_angles = angles.shape[1]

    J_Comp = np.zeros([number_of_modes, number_of_angles], dtype=complex)
    l_ind = 0
    for l in range(ell_max + 1):
        for m in range(-l, l + 1):
            #            print(m, l, m+l+l_ind)
            J_Comp[m + l + l_ind, :] = sph_harm(m, l, angles[1, :] + np.pi, angles[0, :] + np.pi / 2)
            # J_Comp[m + l + l_ind,:] = sph_harm(m, l, angles[1,:]+np.pi, angles[0,:])
        l_ind = l_ind + 2 * l + 1
    return J_Comp


def U(l, m, n, R_1, R_lm1):
    return P(0, l, m, n, R_1, R_lm1)


def V(l, m, n, R_1, R_lm1):
    if (m == 0):
        p0 = P(1, l, 1, n, R_1, R_lm1)
        p1 = P(-1, l, -1, n, R_1, R_lm1)
        ret = p0 + p1
    else:
        if (m > 0):
            d = (m == 1)
            p0 = P(1, l, m - 1, n, R_1, R_lm1)
            p1 = P(-1, l, -m + 1, n, R_1, R_lm1)
            ret = p0 * np.sqrt(1 + d) - p1 * (1 - d)
        else:
            d = (m == -1)
            p0 = P(1, l, m + 1, n, R_1, R_lm1)
            p1 = P(-1, l, -m - 1, n, R_1, R_lm1)
            ret = p0 * (1 - d) + p1 * np.sqrt(1 + d)
    return ret


def Wf(l, m, n, R_1, R_lm1):
    if (m == 0):
        error('should not be called')
    else:
        if (m > 0):
            p0 = P(1, l, m + 1, n, R_1, R_lm1)
            p1 = P(-1, l, -m - 1, n, R_1, R_lm1)
            ret = p0 + p1
        else:
            p0 = P(1, l, m - 1, n, R_1, R_lm1)
            p1 = P(-1, l, -m + 1, n, R_1, R_lm1)
            ret = p0 - p1
    return ret


def P(i, l, a, b, R_1, R_lm1):
    ri1 = R_1[i + 1, 1 + 1]
    rim1 = R_1[i + 1, -1 + 1]
    ri0 = R_1[i + 1, 0 + 1]

    if (b == -l):
        ret = ri1 * R_lm1[a + l - 1, 0] + rim1 * R_lm1[a + l - 1, 2 * l - 2]
    else:
        if (b == l):
            ret = ri1 * R_lm1[a + l - 1, 2 * l - 2] - rim1 * R_lm1[a + l - 1, 0]
        else:
            ret = ri0 * R_lm1[a + l - 1, b + l - 1]
    return ret


def Real_Rotation(rotation, ell_max):
    R = np.zeros([(ell_max + 1) ** 2, (ell_max + 1) ** 2])

    # Trivial rep
    R[0, 0] = 1

    # First band, directly relates to rotation matrix
    R[1, 1] = rotation[1, 1]
    R[1, 2] = rotation[1, 2]
    R[1, 3] = rotation[1, 0]
    R[2, 1] = rotation[2, 1]
    R[2, 2] = rotation[2, 2]
    R[2, 3] = rotation[2, 0]
    R[3, 1] = rotation[0, 1]
    R[3, 2] = rotation[0, 2]
    R[3, 3] = rotation[0, 0]

    R_1 = R[1:4, 1:4]
    R_lm1 = R_1

    # For each subsequent band we progress recursively
    band_idx = 4
    for l in range(2, ell_max + 1):
        R_l = np.zeros([2 * l + 1, 2 * l + 1])
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):

                d = (m == 0)
                if np.abs(n) == l:
                    denom = 2 * l * (2 * l - 1)
                else:
                    denom = l * l - n * n

                u = np.sqrt((l * l - m * m) / denom)
                v = np.sqrt((1 + d) * (l + np.abs(m) - 1) * (l + np.abs(m)) / denom) * (1 - 2 * d) * 0.5
                w = np.sqrt((l - np.abs(m) - 1) * (l - np.abs(m)) / denom) * (1 - d) * (-0.5)

                if u != 0:
                    u = u * U(l, m, n, R_1, R_lm1)
                if v != 0:
                    v = v * V(l, m, n, R_1, R_lm1)
                if w != 0:
                    w = w * Wf(l, m, n, R_1, R_lm1)

                R_l[m + l, n + l] = u + v + w;
        R[band_idx:band_idx + 2 * l + 1, band_idx:band_idx + 2 * l + 1] = R_l;
        R_lm1 = R_l;
        band_idx = band_idx + 2 * l + 1;
    return R


def irrep_transforms_sphere(ell_max, rand_rotations):
    R = np.zeros([rand_rotations.shape[0], (ell_max + 1) ** 2, (ell_max + 1) ** 2])
    for rotation in range(rand_rotations.shape[0]):
        R[rotation, :, :] = Real_Rotation(rand_rotations[rotation, :, :], ell_max)
    return R