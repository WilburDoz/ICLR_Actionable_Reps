# Helper functions for neural representation theory
import os
from datetime import datetime 
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pickle

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

def save_obj(obj, name, savepath):
    with open(savepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, filepath):
    with open(filepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)

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