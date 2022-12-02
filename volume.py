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

T = 150000                   # How many gradient steps
D = 49                        # How many neurons
K = 5                       # How many repeats to run at once
N_rand = 150                # How many random angles, to use for separation loss
N_shift = 15                # Number of other rooms to measure positivity and norm
Shift_std = 4               # Standard deviation of normal from which to sample shifts
norm_size = 1               # How much bigger to make the room you take the norm over
sampling_choice = 1         # 0 for square room, 1 for normal distribution, 2 for circular room
shift_points_sep = 0        # 0 for room centered on (0,0), 1 for room whose centre shifts by shift_std every step
resample_iters = 5          # How often to resample random points

# Set of parameters for the positivity geco
lambda_pos_init = 0.1         # 15 for euc, 5 for kern, 150 for euc_A (maybe 0.5 after N), 0.1 for kern_A, 0.05 BEFORE  # Initial positivity loss weighting
k_p = -9                    # Positivity target
alpha_p = 0.9               # Smoothing of positivity dynamics
gamma_p = 0.0001             # Proportionality constant

# Norm GECO parameters
lambda_norm_init = 0.005      # Initial norm loss weighting
k_norm = 4                   # norm target
alpha_norm = 0.9             # Smoothing of norm dynamics
gamma_norm = 0.0001         # Proportionality constant from mismatch to constrant movement

# Parameters for ADAM
epsilon_w = 0.1             # Step size parameter W
epsilon_om = 0.1            # Frequency step size
beta1 = 0.9                 # Exp moving average parameter for first moment
beta2 = 0.9                 # exp moving average parameter for second moment
eta = 1e-8                   # Small regularising, non-exploding thingy, not v important it seems

# Printing and saving
save_iters = 5               # How often to save results
print_iters = 250            # How often to print results
# save_traj = 1              # Do you want to save a history of w and om?

# Create and save parameter dict
parameters = {"D": D, "T": T, "K": K, "N_rand": N_rand, "N_shift": N_shift, "resample_iters": resample_iters, "save_iters": save_iters,
              "lambda_pos_init": lambda_pos_init, "k_p": k_p, "alpha_p": alpha_p, "gamma_p": gamma_p,
              "lambda_norm_init": lambda_norm_init, "k_norm": k_norm, "alpha_norm": alpha_norm, "gamma_norm": gamma_norm,
              "beta1": beta1, "beta2": beta2, "eta": eta, "epsilon_w": epsilon_w, "epsilon_om": epsilon_om, "dim": 2,
              "sampling_choice": sampling_choice, "norm_size": norm_size, "Shift_std": Shift_std, "shift_points_sep": shift_points_sep}

# Frequency choices
# 0: just all the frequencies up to M, requires equivariance enforcing
# 1: square or rectangular lattice at some base frequency up to (D-1)/2 of them
# 1.5: An approximation to any 2D lattice, as close as you can get on integer lattice
# 2: some random set of (D-1)/2 frequencies
# 3: A sweep through a load of frequency grids
# 4: a pair of grids
# 5: a tuneable set of grid modules
# 6: search through some set of pairs of modules
om_init_scheme = 0
W_constrain = 0
M = int(np.floor((D - 1) / 2))
if om_init_scheme == 0:
    om_init_scale = 2  # om init scale

    parameters.update({"om_init_scheme": om_init_scheme, "om_init_scale": om_init_scale})
else:
    equi_flag = 0
    if om_init_scheme == 1:
        base_freqs = np.array([1, 1])           # Base frequencies, integers

        parameters.update({"om_init_scheme": om_init_scheme, "base_freqs": base_freqs})
        om = helper_functions.freq_selector(M)
        om = np.multiply(om, base_freqs[None, :])
    elif om_init_scheme == 1.5:
        base_lengthscale = 1.3
        relative_scale = 1
        relative_angle = np.pi/3

        parameters.update({"om_init_scheme": om_init_scheme, "base_lengthscale": base_lengthscale, "relative_scale": relative_scale, "relative_angle": relative_angle})
        om = helper_functions.freqs_grid_plane(base_lengthscale, relative_scale, relative_angle, M)
    elif om_init_scheme == 2:
        max_freq = 30

        parameters.update({"om_init_scheme": om_init_scheme, "max_freq": max_freq})
        om = np.random.randint(1, max_freq, size=[M, 2])
    elif om_init_scheme == 3:
        base_min = 0.5
        base_max = 2
        base_step = 0.5
        relative_min = 1 # always greater than 1!
        relative_max = 2
        num_angles = 5
        angles = np.linspace(np.pi/3, np.pi/2, num_angles)

        grid_params = np.meshgrid(np.arange(base_min, base_max+1, base_step), np.arange(relative_min, relative_max+1), angles)
        K = grid_params[0].size          # How many harmonics to try
        parameters.update({"K": K, "base_min": base_min, "base_max": base_max, "relative_min": relative_min, "relative_max": relative_max, "num_angles": num_angles})

        base_freqs = np.ndarray.flatten(grid_params[0])
        relative_freqs = np.ndarray.flatten(grid_params[1])
        angles = np.ndarray.flatten(grid_params[2])
    elif om_init_scheme == 4:
        proportion = 0.65
        base_lengthscale_1 = 1.3
        relative_scale_1 = 1
        relative_angle_1 = np.pi/3
        base_lengthscale_2 = 1.3*1.4
        relative_scale_2 = 1
        relative_angle_2 = np.pi/3
        module_angle = 10/180*np.pi
        W_constrain = 1

        M_1 = int(M*proportion)
        M_2 = M - M_1

        parameters.update({"om_init_scheme": om_init_scheme, "proportion": proportion, "base_lengthscale_1": base_lengthscale_1,
                           "relative_scale_1": relative_scale_1, "relative_angle_1": relative_angle_1, "base_lengthscale_2": base_lengthscale_2,
                           "relative_scale_2": relative_scale_2, "relative_angle_2": relative_angle_2, "module_angle": module_angle,
                           "W_constrain": W_constrain})
        om_1 = helper_functions.freqs_grid_plane(base_lengthscale_1, relative_scale_1, relative_angle_1, M_1)
        om_2 = helper_functions.freqs_grid_plane(base_lengthscale_2, relative_scale_2, relative_angle_2, M_2, module_angle)
        om = np.vstack([om_1, om_2])

        if W_constrain:
            mask = np.zeros([D, D])
            mask[1:M_1 + 1, M_1 + 1:] = 1
            mask = mask + np.transpose(mask)
    elif om_init_scheme == 5:
        Q = 2               # Number of modules
        mod_init_scale = 2  # normal to draw the grid params from
        M_Q = int(M/Q)

        parameters.update({"Q": Q, "om_init_scheme": om_init_scheme, "mod_init_scale": mod_init_scale})
    elif om_init_scheme == 6:
        # here these will be fixed
        base_lengthscale_1 = 1.3
        relative_scale_1 = 1
        relative_angle_1 = np.pi/3
        relative_scale_2 = 1
        relative_angle_2 = np.pi/3
        proportion = 0.65
        M_1 = int(M*proportion)
        M_2 = M - M_1
        W_constrain = 1

        # and we'll search through
        base_lengthscale_2 = base_lengthscale_1*np.linspace(1.1, 1.6, 5)
        module_angle = np.linspace(3, 30, 10)*np.pi/180
        counter = 0
        comb_param = np.zeros([10*5, 2])
        for base in base_lengthscale_2:
            for module in module_angle:
                comb_param[counter, :] = [base, module]
                counter += 1

        K = 5*10

# Separation loss choices
# 0: Simple euclidean loss
# 1: Chi weighted euclidean
# 2: Kernel loss
# 3: Chi weighted kernel loss
sep_loss_choice = 3
# Chi choices
# 0: Standard inverted gaussian thing
# 1: Just euclidean growing!
# 2: Exponential growing
chi_choice = 0
if sep_loss_choice == 0:
    loss_sep = jit(losses.sep_plane_Euc)
    grad_sep_W = jit(grad(losses.sep_plane_Euc, argnums=0))
    grad_sep_om = jit(grad(losses.sep_plane_Euc, argnums=1))
elif sep_loss_choice == 1:
    sigma_theta = 1
    f = 1

    parameters.update({"sigma_theta": sigma_theta, "f": f, "chi_choice": chi_choice})
    loss_sep = jit(losses.sep_plane_EucChi)
    grad_sep_W = jit(grad(losses.sep_plane_EucChi, argnums=0))
    grad_sep_om = jit(grad(losses.sep_plane_EucChi, argnums=1))
    if chi_choice == 0:
        calc_chi = jit(helper_functions.calc_chi_plane)
    elif chi_choice == 1:
        calc_chi = jit(helper_functions.calc_chi_plane_euc)
elif sep_loss_choice == 2:
    sigma_sq = 0.05

    parameters.update({"sigma_sq": sigma_sq})
    loss_sep = jit(losses.sep_plane_Kern)
    grad_sep_W = jit(grad(losses.sep_plane_Kern, argnums=0))
    grad_sep_om = jit(grad(losses.sep_plane_Kern, argnums=1))
elif sep_loss_choice == 3:
    sigma_sq = 0.4
    sigma_theta = 1
    f = 1

    parameters.update({"sigma_sq": sigma_sq, "sigma_theta": sigma_theta, "f": f, "chi_choice": chi_choice})
    if om_init_scheme == 5:
        loss_sep = jit(losses.sep_plane_KernChi_Module)
        grad_sep_W = jit(grad(losses.sep_plane_KernChi_Module, argnums=0))
        grad_sep_om = jit(grad(losses.sep_plane_KernChi_Module, argnums = 1))
    else:
        loss_sep = jit(losses.sep_plane_KernChi)
        grad_sep_W = jit(grad(losses.sep_plane_KernChi, argnums=0))
        grad_sep_om = grad(losses.sep_plane_KernChi, argnums=1)
    if chi_choice == 0:
        calc_chi = jit(helper_functions.calc_chi_plane)
    elif chi_choice == 1:
        calc_chi = jit(helper_functions.calc_chi_plane_euc)
    elif chi_choice == 2:
        calc_chi = jit(helper_functions.calc_chi_plane_exp)

parameters.update({"sep_loss_choice": sep_loss_choice})
loss_pos = jit(losses.pos_plane)
grad_pos_W = jit(grad(losses.pos_plane, argnums=0))
grad_pos_om = jit(grad(losses.pos_plane, argnums=1))
loss_norm = jit(losses.norm_plane)
grad_norm_W = jit(grad(losses.norm_plane, argnums=0))
grad_norm_om = jit(grad(losses.norm_plane, argnums=1))
init_irreps = jit(helper_functions.init_irreps_2D)
key = random.PRNGKey(0)

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
    # Randomly initialise the weights, losses, moments, and best W and loss
    key, subkey1 = random.split(key)     # How to do random things in jax
    W = random.normal(subkey1, [D, 2*M+1])    # The random init weights
    if om_init_scheme == 0:
        key, subkey2 = random.split(key)
        om = random.uniform(subkey2, [M, 3]) * om_init_scale
    elif om_init_scheme == 3:
        om = helper_functions.freqs_grid_plane(base_freqs[counter], relative_freqs[counter], angles[counter], M)
    elif om_init_scheme == 5:
        om = np.random.normal(0, mod_init_scale, [Q*4])
    elif om_init_scheme == 6:
        om_1 = helper_functions.freqs_grid_plane(base_lengthscale_1, relative_scale_1, relative_angle_1, M_1)
        om_2 = helper_functions.freqs_grid_plane(comb_param[counter, 0], relative_scale_2, relative_angle_2, M_2, comb_param[counter, 1])
        om = np.vstack([om_1, om_2])

    W_init = W
    means_W = jnp.zeros(jnp.shape(W))     # Moments for ADAM
    sec_moms_W = jnp.zeros(jnp.shape(W))
    W_best = W                          # Initialise best W somewhere

    om_init = om
    means_om = jnp.zeros(jnp.shape(om))  # Moments for ADAM
    sec_moms_om = jnp.zeros(jnp.shape(om))
    om_best = om

    Losses = np.zeros([4, int(T / save_iters)])   # Holder for losses, total, sep, and equi
    min_L = np.zeros([5])  # Step, Loss, Loss_Sep, and Loss_Equi at min Loss
    min_L[1] = np.infty                 # Set min Loss = infty
    L2 = 0                              # So that the positivity moving average has somewhere to start
    L3 = 0                              # Starting norm average
    lambda_norm = lambda_norm_init      # Starting lambda norm
    lambda_pos = lambda_pos_init        # And the positivity
    save_counter = 0

    #if save_traj:
    #    W_hist = np.zeros([W.shape[:], int(T / save_iters)])
    #    om_hist = np.zeros([om.shape[:], int(T / save_iters)])

    for step in range(T):
        if step % resample_iters == 0:
            # Create the angles, shifts, irreps, and transforms
            if sampling_choice == 0:
                phi = (np.random.sample([N_rand, 3]) - 0.5) * np.pi * 2
            elif sampling_choice == 1:
                phi = np.random.normal(0, 1, [N_rand, 3])
            elif sampling_choice == 2:
                r = np.sqrt(np.random.uniform(size=[N_rand, 1]))
                theta = np.random.uniform(size=[N_rand, 1]) * 2 * np.pi
                phi = np.hstack([r * np.cos(theta), r * np.sin(theta)])

            phi_shift = np.random.normal(0, Shift_std, [N_shift, 3])
            # phi_shift = (np.random.randint(1 + 2*Shift_std, size=N_shift) - Shift_std)*2 * np.pi
            phi_norm = norm_size*np.reshape(phi[:, None, :] + phi_shift[None, :, :], [N_rand * N_shift, 3], order='F')
            phi_pos = np.vstack([phi, phi_norm])

            if shift_points_sep:
                phi = phi + phi_shift[np.random.randint(N_shift), :][None, :]

            if sep_loss_choice == 1 or sep_loss_choice == 3:
                chi = calc_chi(phi, sigma_theta, f)

        # Separation Term
        if sep_loss_choice == 0:
            L1 = loss_sep(W, om, phi)
            W_grad1 = grad_sep_W(W, om, phi)
            if om_init_scheme == 0:
                om_grad1 = grad_sep_om(W, om, phi)
        elif sep_loss_choice == 1:
            L1 = loss_sep(W, om, phi, chi)
            W_grad1 = grad_sep_W(W, om, phi, chi)
            if om_init_scheme == 0:
                om_grad1 = grad_sep_om(W, om, phi, chi)
        elif sep_loss_choice == 2:
            L1 = loss_sep(W, om, phi, sigma_sq)
            W_grad1 = grad_sep_W(W, om, phi, sigma_sq)
            if om_init_scheme == 0:
                om_grad1 = grad_sep_om(W, om, phi, sigma_sq)
        elif sep_loss_choice == 3:
            L1 = loss_sep(W, om, phi, sigma_sq, chi)
            W_grad1 = grad_sep_W(W, om, phi, sigma_sq, chi)
            if om_init_scheme == 0 or om_init_scheme == 5:
                om_grad1 = grad_sep_om(W, om, phi, sigma_sq, chi)

        # Positivity Term
        pos = loss_pos(W, om, phi_pos, N_shift)
        W_grad2 = grad_pos_W(W, om, phi_pos, N_shift)
        if om_init_scheme == 0 or om_init_scheme == 5:
            om_grad2 = grad_pos_om(W, om, phi_pos, N_shift)
        if pos > 0:
            L2_Here = np.log(pos) - k_p
            if L2_Here > 0:
                L2_Here = np.log(L2_Here)
        else:
            L2_Here = -5
        L2 = L2*alpha_p + (1 - alpha_p)*L2_Here
        lambda_pos = lambda_pos*np.exp(L2*gamma_p)

        L3_Here = np.log(loss_norm(W, om, phi, phi_norm)) - k_norm
        L3 = L3 * alpha_norm + (1 - alpha_norm) * L3_Here
        lambda_norm = lambda_norm * np.exp(L3 * gamma_norm)
        W_grad3 = grad_norm_W(W, om, phi, phi_norm)
        if om_init_scheme == 0 or om_init_scheme == 5:
            om_grad3 = grad_norm_om(W, om, phi, phi_norm)

        # Update the moment averages, then bias correct them
        W_grad = W_grad1 + lambda_pos*W_grad2 + lambda_norm*W_grad3
        means_W = beta1*means_W + (1 - beta1)*W_grad
        sec_moms_W = beta2*sec_moms_W + (1 - beta2)*np.power(W_grad, 2)
        means_debiased_W = means_W/(1 - np.power(beta1, step+1))
        sec_moms_debiased_W = sec_moms_W/(1 - np.power(beta2, step + 1))

        if om_init_scheme == 0 or om_init_scheme == 5:
            om_grad = om_grad1 + lambda_pos * om_grad2 + lambda_norm * om_grad3
            means_om = beta1 * means_om + (1 - beta1) * om_grad
            sec_moms_om = beta2 * sec_moms_om + (1 - beta2) * np.power(om_grad, 2)
            means_debiased_om = means_om / (1 - np.power(beta1, step + 1))
            sec_moms_debiased_om = sec_moms_om / (1 - np.power(beta2, step + 1))

        if step % save_iters == 0:        # Save and print the appropriate losses
            if L2 > 0:
                Losses[0, save_counter] = L1 + L2 * lambda_pos
            else:
                Losses[0, save_counter] = L1
            if L3 > 0:
                Losses[0, save_counter] += L3 * lambda_norm
            Losses[1, save_counter] = L1
            Losses[2, save_counter] = L2_Here
            Losses[3, save_counter] = L3_Here
            save_counter = save_counter + 1

        if step % print_iters == 0:
            print(f'Iteration: {step}, Loss: {Losses[1, save_counter-1]:.5f}\t Sep: {L1:.5f}\t Pos: {L2_Here:.5f}\t {L2:.5f}\t L P: {lambda_pos:.5f}\t Norm: {L3_Here:.5f}\t {L3:.5f}\t L N: {lambda_norm:.5f}')

        # Potentially save the best results
        if Losses[1, save_counter-1] < min_L[1] and L2 <= 0 and L3 < 0:
            min_L = [save_counter-1, Losses[0, save_counter-1], Losses[1, save_counter-1], Losses[2, save_counter-1]]
            W_best = W
            om_best = om

        # Take parameter step
        W = W - epsilon_w*means_debiased_W/(np.sqrt(sec_moms_debiased_W + eta))
        if om_init_scheme == 0 or om_init_scheme == 5:
            om = om - epsilon_om * means_debiased_om / (np.sqrt(sec_moms_debiased_om + eta))
        if W_constrain:
            W = W.at[:M_1*2 + 2, M_1*2 + 1:].set(0)
            W = W.at[M_1*2 + 2:, 1:M_1*2 + 1].set(0)

    # Now save the weights and the losses
    helper_functions.save_obj(W_best, f"W_{counter}", savepath)
    helper_functions.save_obj(W_init, f"W_init_{counter}", savepath)
    helper_functions.save_obj(Losses, f"L_{counter}", savepath)
    helper_functions.save_obj(min_L, f"min_L_{counter}", savepath)
    helper_functions.save_obj(om, f"om_{counter}", savepath)
    helper_functions.save_obj(om_best, f"om_{counter}", savepath)
    helper_functions.save_obj(W, f"W_final_{counter}", savepath)
    helper_functions.save_obj(om, f"om_final_{counter}", savepath)

    # And print to say iteration done
    print(f"\nDONE ITERATION {counter}: Min_Loss = {min_L[1]:.5f}\n")
