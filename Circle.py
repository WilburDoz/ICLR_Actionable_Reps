# Load up packages
from jax import grad, jit, random
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os

# And functions I've written
from NRT_functions import helper_functions
from NRT_functions import losses

##### Set a load of parameters ######

T = 50000                   # How many gradient steps
D = 21                      # How many neurons
K = 5                       # How many repeats to run at once
N_rand = 150                # How many random angles, to use for separation loss
N_pos = 450                 # How many angles to use for positivity
resample_iters = 10         # How often to resample random points

# Set of parameters for the positivity geco
lambda_pos_init = 1         # Good for kern 0.1 # Good for euc 1 # Initial positivity loss weighting
k_p = -9                    # Positivity target
alpha_p = 0.9               # Smoothing of positivity dynamics
gamma_p = 0.0001           # Proportionality constant

# Parameters for ADAM
epsilon = 0.01             # Step size parameter W
beta1 = 0.9                 # Exp moving average parameter for first moment
beta2 = 0.9                 # exp moving average parameter for second moment
eta = 1e-8                  # Small regularising, non-exploding thingy, not v important it seems

# Printing and saving
save_iters = 5  # How often to save results
print_iters = 250  # How often to print results

# Create and save parameter dict
parameters = {"D": D, "T": T, "K": K, "N_rand": N_rand, "N_pos": N_pos, "resample_iters":resample_iters, "save_iters": save_iters,
              "lambda_pos_init": lambda_pos_init, "k_p": k_p, "alpha_p": alpha_p, "gamma_p": gamma_p,
              "beta1": beta1, "beta2": beta2, "eta": eta, "epsilon": epsilon, "dim": 1}

# Frequency choices
# 0: just all the frequencies up to M, requires equivariance enforcing
# 1: all the harmonics of up to (D-1)/2 of some base freq
# 2: some random set of (D-1)/2 frequencies
# 3: all the harmonics at a base frequency that increases slowly
om_init_scheme = 0
if om_init_scheme == 0:
    M = 100                     # Number of frequencies
    N_shift = 3                 # Number of shifts to check in equi loss

    # Equivariance GECO parameters
    lambda_equi_init = 0.0001        # Initial equivariance loss weighting
    k_eq = -5                   # Equivaraince target
    alpha_eq = 0.9              # Smoothing of equivariance dynamics
    gamma_eq = 0.0001           # Proportionality constant from mismatch to constrant movement

    parameters.update({"om_init_scheme":om_init_scheme,"M": M, "N_shift": N_shift,
                       "lambda_equi_init": lambda_equi_init, "k_eq": k_eq, "alpha_eq": alpha_eq, "gamma_eq": gamma_eq})
    om = np.arange(1, M+1)
    loss_equi = jit(losses.equi_circ_smart_B)
    grad_equi = jit(grad(losses.equi_circ_smart_B, argnums=0))
    grad_equi_B = jit(grad(losses.equi_circ_smart_B, argnums=1))
    #transforms = jit(helper_functions.irrep_transforms_1D) # jiting it doesn't work, would have to rewrite cleverly...
    equi_flag = 1
else:
    equi_flag = 0
    if om_init_scheme == 1:
        base_freq = 3           # Base frequency

        parameters.update({"om_init_scheme":om_init_scheme,"base_freq":base_freq})
        M = int(np.floor((D - 1) / 2))
        om = np.arange(1, M+1)*base_freq
    elif om_init_scheme == 2:
        max_freq = 30

        parameters.update({"om_init_scheme":om_init_scheme,"max_freq":max_freq})
        M = int(np.floor((D - 1) / 2))
        om = np.random.randint(1, max_freq, size=M)
    elif om_init_scheme == 3:
        M = int(np.floor((D - 1) / 2))
        om_harm = np.arange(1, M+1)

        K = 10          # How many harmonics to try
        parameters.update({"K":K})

# Separation loss choices
# 0: Simple euclidean loss
# 1: Chi weighted euclidean
# 2: Kernalised loss
# 3: Kernelised chi weighted
sep_loss_choice = 0
if sep_loss_choice == 0:
    loss_sep = jit(losses.sep_circ_Euc)
    grad_sep = jit(grad(losses.sep_circ_Euc, argnums=0))
elif sep_loss_choice == 1:
    sigma_theta = 0.1
    f = 1

    parameters.update({"sigma_theta": sigma_theta, "f": f})
    loss_sep = jit(losses.sep_circ_EucChi)
    grad_sep = jit(grad(losses.sep_circ_EucChi, argnums=0))
    calc_chi = jit(helper_functions.calc_chi_circ)
elif sep_loss_choice == 2:
    sigma_sq = 7

    parameters.update({"sigma_sq": sigma_sq})
    loss_sep = jit(losses.sep_circ_Kern)
    grad_sep = jit(grad(losses.sep_circ_Kern, argnums=0))
elif sep_loss_choice == 3:
    sigma_sq = 2
    sigma_theta = 0.25
    f = 1

    parameters.update({"sigma_sq": sigma_sq, "sigma_theta": sigma_theta, "f": f})
    loss_sep = jit(losses.sep_circ_KernChi)
    grad_sep = jit(grad(losses.sep_circ_KernChi, argnums=0))
    calc_chi = jit(helper_functions.calc_chi_circ)

# How to sample all the points
# 0: Uniform on circle
# 1: Separation loss from von mises, positivity all around
sample_choice = 0
if sample_choice == 1:
    spread = 0.1
    parameters.update({"spread": spread})

parameters.update({"sep_loss_choice": sep_loss_choice, "sample_choice": sample_choice})
loss_pos = jit(losses.pos_circ)
grad_pos = jit(grad(losses.pos_circ, argnums = 0))
init_irreps = jit(helper_functions.init_irreps_1D)
key = random.PRNGKey(0)

# Setup save file locations
today =  datetime.strftime(datetime.now(),'%y%m%d')
now = datetime.strftime(datetime.now(),'%H%M%S')
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
    # Randomly initialise the weights, losses, moments, and best W and loss
    key, subkey1 = random.split(key)     # How to do random things in jax
    W = random.normal(subkey1,[D,2*M+1])    # The random init weights
    W_init = W
    means_W = jnp.zeros(jnp.shape(W))     # Moments for ADAM
    sec_moms_W = jnp.zeros(jnp.shape(W))
    W_best = W                          # Initialise best W somewhere

    if om_init_scheme == 3:
        om = (counter + 1)*om_harm

    if equi_flag:
        Losses = np.zeros([4, int(T / save_iters)+1])   # Holder for losses, total, sep, and equi
        min_L = np.zeros([5])  # Step, Loss, Loss_Sep, and Loss_Equi at min Loss
        key, subkey2 = random.split(key)
        B = random.normal(subkey2,[2*M+1,D])
        B_init = B
        B_best = B
        means_B = jnp.zeros(jnp.shape(B))     # Moments for ADAM
        sec_moms_B = jnp.zeros(jnp.shape(B))
        lambda_equi = lambda_equi_init      # Initialise the starting lambda_equi
    else:
        Losses = np.zeros([3, int(T / save_iters)])   # Holder for losses, total, sep, and equi
        min_L = np.zeros([4])  # Step, Loss, Loss_Sep, and Loss_Equi at min Loss

    min_L[1] = np.infty                 # Set min Loss = infty
    L2 = 0                              # So that the positivity moving average has somewhere to start
    L3 = 0                              # Same for the equivariance
    lambda_pos = lambda_pos_init        # And the positivity
    save_counter = 0

    Losses[0,:] = np.infty

    for step in range(T):
        if step%resample_iters == 0:
            # Create the angles, shifts, irreps, and transforms
            if sample_choice == 0:
                phi = 2*np.pi*np.random.sample(N_rand)
            else:
                phi = np.random.sample(N_rand)*spread
            I = init_irreps(om, phi)

            phi_pos = 2*np.pi*np.random.sample(N_pos)

            if equi_flag:
                phi_shift = np.hstack([0, 2 * np.pi * np.random.sample(N_shift)])
                phi_full = np.mod(np.ndarray.flatten(phi_pos[:, np.newaxis] + phi_shift[np.newaxis, :], order='F'), 2 * np.pi)
                I_full = init_irreps(om, phi_full)
                I_pos = I_full[:, :N_pos]
                G_I = helper_functions.irrep_transforms_1D(om, phi_shift[1:])
            else:
                I_pos = init_irreps(om, phi_pos)

            if sep_loss_choice == 1 or sep_loss_choice == 3:
                chi = calc_chi(phi, sigma_theta, f)



        # Separation Term
        if sep_loss_choice == 0:
            L1 = loss_sep(W, I)
            W_grad1 = grad_sep(W, I)
        elif sep_loss_choice == 1:
            L1 = loss_sep(W, I, chi)
            W_grad1 = grad_sep(W, I, chi)
        elif sep_loss_choice == 2:
            L1 = loss_sep(W, I, sigma_sq)
            W_grad1 = grad_sep(W, I, sigma_sq)
        elif sep_loss_choice == 3:
            L1 = loss_sep(W, I, sigma_sq, chi)
            W_grad1 = grad_sep(W, I, sigma_sq, chi)

        # Positivity Term
        pos = loss_pos(W, I_pos)
        W_grad2 = grad_pos(W, I_pos)
        if pos > 0:
            L2_Here = np.log(pos) - k_p
            if L2_Here > 0:
                L2_Here = np.log(L2_Here)
        else:
            L2_Here = -5
        L2 = L2*alpha_p + (1 - alpha_p)*L2_Here
        lambda_pos = lambda_pos*np.exp(L2*gamma_p)

        if equi_flag:           # Equivariance term
            L3_Here = np.log(loss_equi(W, B, I_full[:,:N_pos], I_full[:,N_pos:], G_I)) - k_eq
            L3 = L3*alpha_eq + (1 - alpha_eq)*L3_Here
            lambda_equi = lambda_equi*np.exp(L3*gamma_eq)
            W_grad3 = grad_equi(W, B, I_full[:, :N_pos], I_full[:, N_pos:], G_I)
            B_grad = grad_equi_B(W, B, I_full[:, :N_pos], I_full[:, N_pos:], G_I)

        # Update the moment averages, then bias correct them
        W_grad = W_grad1 + lambda_pos*W_grad2
        if equi_flag:
            W_grad = W_grad + lambda_equi*W_grad3

            means_B = beta1 * means_B + (1 - beta1) * B_grad
            sec_moms_B = beta2 * sec_moms_B + (1 - beta2) * np.power(B_grad, 2)
            means_debiased_B = means_B / (1 - np.power(beta1, step + 1))
            sec_moms_debiased_B = sec_moms_B / (1 - np.power(beta2, step + 1))

        means_W = beta1*means_W + (1 - beta1)*W_grad
        sec_moms_W = beta2*sec_moms_W + (1 - beta2)*np.power(W_grad,2)
        means_debiased_W = means_W/(1 - np.power(beta1, step+1))
        sec_moms_debiased_W = sec_moms_W/(1 - np.power(beta2, step + 1))


        if step%save_iters == 0:        # Save and print the appropriate losses
            if L2 > 0:
                Losses[0, save_counter] = L1 + L2 * lambda_pos
            else:
                Losses[0, save_counter] = L1
            if equi_flag and L3 > 0:
                Losses[0, save_counter] = Losses[0, save_counter] + L3 * lambda_pos
                Losses[3, save_counter] = L3_Here
            Losses[1, save_counter] = L1
            Losses[2, save_counter] = L2_Here

            # Potentially save the best results
            if Losses[0, save_counter] < min_L[1] and L2_Here <= 0:
                if equi_flag and L3_Here < 0:
                    min_L = [save_counter, Losses[0, save_counter], Losses[1, save_counter], Losses[2, save_counter],
                             Losses[3, save_counter]]
                    B_best = B
                else:
                    min_L = [save_counter, Losses[0, save_counter], Losses[1, save_counter], Losses[2, save_counter]]
                W_best = W

            save_counter = save_counter + 1

        if step%print_iters == 0:
            if equi_flag:
                print(f'Iteration: {step}, Loss: {Losses[0, save_counter-1]:.5f}\t Sep: {L1:.5f}\t Equ: {L3_Here:.5f}\t L Eq: {lambda_equi:.5f}\t Pos: {L2_Here:.5f}\t L P: {lambda_pos:.5f}')
            else:
                print(f'Iteration: {step}, Loss: {Losses[0, save_counter-1]:.5f}\t Sep: {L1:.5f}\t Pos: {L2_Here:.5f}\t L P: {lambda_pos:.5f}')

        # Take parameter step
        W = W - epsilon*means_debiased_W/(np.sqrt(sec_moms_debiased_W + eta))
        if equi_flag:
            B = B - epsilon*means_debiased_B/(np.sqrt(sec_moms_debiased_B + eta))

    # Now save the weights and the losses
    W_best = helper_functions.normalise_weights(W_best)
    W_init = helper_functions.normalise_weights(W_init)
    helper_functions.save_obj(W_best, f"W_{counter}", savepath)
    helper_functions.save_obj(W, f"W_final_{counter}", savepath)
    helper_functions.save_obj(W_init, f"W_init_{counter}", savepath)
    helper_functions.save_obj(Losses, f"L_{counter}", savepath)
    helper_functions.save_obj(min_L, f"min_L_{counter}", savepath)
    helper_functions.save_obj(om, f"om_{counter}", savepath)
    if equi_flag:
        helper_functions.save_obj(B_best, f"B_{counter}", savepath)
        helper_functions.save_obj(B_init, f"B_init_{counter}", savepath)

    # And print to say iteration done
    print(f"\nDONE ITERATION {step}: Min_Loss = {min_L[1]:.5f}\n")