from jax import grad, jit, random
import jax.numpy as jnp
import numpy as np
import argparse
from datetime import datetime
import os

from jax.config import config
config.update("jax_debug_nans", True)

from NRT_functions import losses
from NRT_functions import helper_functions
from scipy.spatial.transform import Rotation as Rot

##### Set a load of parameters ######

T = 30000                  # How many iterations
D = 9                      # How many neurons
K = 1                      # How many repeats
ell_max = 10               # How many spherical harmonics modes (number of irrep dims grows as n^2!)
N_rand = 75                 # How many random angles, to use for separation loss
N_shift = 4                 # How many random shifts, to use for equivariance loss
resample_iters = 12          # How often to resample
random_seed = 41           # Random seed

# Equivariance GECO Parameters
lambda_equi_init = 0.25     # Initial equivariance loss weighting
k_eq = -5                   # Equivaraince target
alpha_eq = 0.95             # Smoothing of equivariance dynamics
gamma_eq = 0.0001            # Proportionality constant from mismatch to constrant movement

# Positivity GECO Parameters
lambda_pos_init = 0.5       # Initial positivity loss weighting
k_p = -6                    # Positivity target
alpha_p = 0.9               # Smoothing of positivity dynamics
gamma_p = 0.0001             # Proportionality constant

# Parameters for ADAM
epsilon = 0.01               # Step size parameter
beta1 = 0.9                 # Exp moving average parameter for first moment
beta2 = 0.9                 # exp moving average parameter for second moment
eta = 1e-8                  # Small regularising, non-exploding thingy, not v important it seems

# Printing and saving
print_iters = 250            # How often to print results

######################################

# Create parameter dict
parameters = {"D": D, "T": T, "K": K, "N_rand": N_rand, "N_shift": N_shift, "ell_max":ell_max, "resample_iters": resample_iters,
              "lambda_pos_init": lambda_pos_init, "k_p": k_p, "alpha_p": alpha_p, "gamma_p": gamma_p,
              "lambda_equi_init": lambda_pos_init, "k_eq": k_p, "alpha_eq": alpha_p, "gamma_eq": gamma_p,
              "beta1": beta1, "beta2": beta2, "eta": eta, "epsilon": epsilon, "random_seed": random_seed}

# Separation Loss Choices:
# 0: Simple euclidean loss
# 1: Chi weighted euclidean
# 2: Kernel loss
# 3: chi weighted kernel
sep_loss_choice = 3
if sep_loss_choice == 3:
    sigma_sq = 1
    sigma_theta = 0.1

    loss_sep = jit(losses.sep_circ_KernChi)
    grad_sep = jit(grad(losses.sep_circ_KernChi))
    calc_chi = jit(helper_functions.calc_chi_sphere)

    parameters.update({"sigma_sq": sigma_sq, "sigma_theta": sigma_theta, "sep_loss_choice": sep_loss_choice})

loss_pos = jit(losses.pos_circ)
grad_pos = jit(grad(losses.pos_circ))
loss_equi = jit(losses.equi_smart)
grad_equi = jit(grad(losses.equi_smart))
key = random.PRNGKey(random_seed)

# Setup save file locations
today = datetime.strftime(datetime.now(), '%y%m%d')
now = datetime.strftime(datetime.now(), '%H%M%S')
filepath = f"./data/{today}/{now}/"
# Make sure folder is there
if not os.path.isdir(f"./data/"):
    os.mkdir(f"./data/")
if not os.path.isdir(f"data/{today}/"):
    os.mkdir(f"data/{today}/")
# Now make a folder in there for this run
savepath = f"data/{today}/{now}/"
if not os.path.isdir(f"data/{today}/{now}"):
    os.mkdir(f"data/{today}/{now}")


helper_functions.save_obj(parameters, "parameters", savepath)
print("\nOPTIMISATION BEGINNING\n")

for counter in range(K):
    # Randomly initialise
    key, subkey = random.split(key)     # How to do random things in jax
    W = random.normal(subkey, [D,(ell_max+1)**2])    # The random init weights
    W_init = W
    means_W = jnp.zeros(jnp.shape(W))     # Moments for ADAM
    sec_moms_W = jnp.zeros(jnp.shape(W))
    W_best = W                          # Initialise best W somewhere

    Losses = np.zeros([4, T])             # Holder for losses, total, sep, and equi
    min_L = np.zeros([5])               # Step, Loss, Loss_Sep, and Loss_Equi at min Loss
    min_L[1] = np.infty                 # Set min Loss = infty
    L2 = 0                              # So that the equivariance moving average has somewhere to start
    L3 = 0                              # Same for the positivity
    lambda_equi = lambda_equi_init      # Initialise the starting lambda_equi
    lambda_pos = lambda_pos_init        # And the positivity
    save_counter = 0

    for step in range(T):
        if step%resample_iters == 0:
            # Sample some base points
            xyz_base = np.random.normal(size = [3,N_rand])
            xyz_base = xyz_base/np.linalg.norm(xyz_base, axis = 0)[np.newaxis, :]
            phi_base = helper_functions.convert_angles(xyz_base)

            # Create some stacked transformed versions
            xyz = np.zeros([3, N_rand*(N_shift+1)])
            phi = np.zeros([2, N_rand*(N_shift+1)])
            xyz[:,:N_rand] = xyz_base
            phi[:,:N_rand] = phi_base
            rand_rotations = Rot.random(N_shift).as_matrix()
            for rand_rot in range(N_shift):
                xyz[:,(rand_rot+1)*N_rand:(rand_rot+2)*N_rand] = np.matmul(rand_rotations[rand_rot,:,:], xyz_base)
                phi[:,(rand_rot+1)*N_rand:(rand_rot+2)*N_rand] = helper_functions.convert_angles(xyz[:,(rand_rot+1)*N_rand:(rand_rot+2)*N_rand])

            I = helper_functions.initialise_irreps_sphere(ell_max, phi)
            G_I = helper_functions.irrep_transforms_sphere(ell_max, rand_rotations)
            if sep_loss_choice == 1 or sep_loss_choice == 3:
                chi = calc_chi(phi[:,:N_rand], sigma_theta)

        # Calculate the losses, and update the equivariance weighting
        L1 = loss_sep(W, I[:,:N_rand], sigma_sq, chi)
        W_grad1 = grad_sep(W, I[:,:N_rand], sigma_sq, chi)

        L2_Here = np.log(loss_equi(W, I[:, :N_rand], I[:, N_rand:], G_I)) - k_eq
        L2 = L2 * alpha_eq + (1 - alpha_eq) * L2_Here
        lambda_equi = lambda_equi*np.exp(L2*gamma_eq)
        W_grad2 = grad_equi(W, I[:,:N_rand], I[:,N_rand:], G_I)

        pos = loss_pos(W, I)
        W_grad3 = grad_pos(W, I)
        if pos > 0:
            L3_Here = np.log(pos) - k_p
            if L3_Here > 0:
                L3_Here = np.log(L3_Here)
        else:
            L3_Here = -5
        L3 = L3*alpha_p + (1 - alpha_p)*L3_Here
        lambda_pos = lambda_pos*np.exp(L3*gamma_p)

        W_grad = W_grad1 +lambda_pos*W_grad3 + lambda_equi*W_grad2 #+ lambda_pos*W_grad3
        means_W = beta1*means_W + (1 - beta1)*W_grad
        sec_moms_W = beta2*sec_moms_W + (1 - beta2)*np.power(W_grad,2)
        means_debiased_W = means_W/(1 - np.power(beta1, step+1))
        sec_moms_debiased_W = sec_moms_W/(1 - np.power(beta2, step + 1))

     # Save and print the appropriate losses
        if L2 > 0:
            Losses[0, step] = L1 + L2*lambda_equi
        else:
            Losses[0, step] = L1
        if L3 > 0:
            Losses[0, step] = Losses[0, step] + L3*lambda_pos
        Losses[1, step] = L1
        Losses[2, step] = L2
        Losses[3, step] = L3_Here
        if step%print_iters == 0:
            print(f'Iteration: {step}, Loss: {Losses[1, step]:.5f}\t Sep: {L1:.5f}\t Equ: {L2_Here:.5f}\t L Eq: {lambda_equi:.5f}\t Pos: {L3_Here:.5f}\t L P: {lambda_pos:.5f}')

        # Potentially save the best results
        if Losses[1, step] < min_L[1] and L2_Here <= 0 and L3_Here < 0:
            min_L = [step, Losses[0, step], Losses[1, step], Losses[2, step], Losses[3,step]]
            W_best = W

        # Take parameter step
        W = W - epsilon * means_debiased_W / (np.sqrt(sec_moms_debiased_W + eta))

    if min_L[1] == np.infty:
        W_best = W

    W_best = helper_functions.normalise_weights(W_best)
    W_init = helper_functions.normalise_weights(W_init)
    helper_functions.save_obj(W_best, f"W_{counter}", savepath)
    helper_functions.save_obj(W_init, f"W_init_{counter}", savepath)
    helper_functions.save_obj(Losses, f"L_{counter}", savepath)
    helper_functions.save_obj(min_L, f"min_L_{counter}", savepath)