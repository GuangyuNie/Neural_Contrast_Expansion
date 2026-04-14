import yaml
import torch
import time
import os
import zipfile
import numpy as np
from helper.microstructure_generation import *
from helper.npcf_calculation import *

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def rotation_matrix(theta):
    """
    Returns the 2D rotation matrix R(theta) for a given angle theta.

    Args:
        theta (float or torch.Tensor): The rotation angle in radians.

    Returns:
        torch.Tensor: A 2x2 rotation matrix.
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    R = torch.tensor([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
    return R
    # return torch.tensor([cos_theta, sin_theta])



def get_T1_T2(file_loc = './total_T_optimized_128_limited.pt',T_w_grad = None, device='cuda'):
    config = load_config('config.yaml')
    sigma0 = config['sigma0']
    sigma1 = config['sigma1']
    size = config['size']
    d = config['d']
    device_id = config['device']
    npcf = config['n']
    gt_size = config['gt_size']
    seed = config['seed']
    mean = config['mean']
    length_scale_x = config['length_scale_x']
    length_scale_y = config['length_scale_y']
    threshold = config['threshold']



    microstructure = get_microstructure(mode='generate', size=64, mean=mean, length_scale_x=length_scale_x,
                                           length_scale_y=length_scale_y, threshold=threshold, seed=seed,
                                           binary=False)
    microstructure = torch.from_numpy(microstructure).float()
    if T_w_grad is not None:
        new_T = T_w_grad
    else:
        new_T = torch.load(file_loc,map_location = device)

    phi = torch.mean(microstructure)
    S2 = twopcf(microstructure)
    S3 = threepcf_fullset(microstructure)

    d = 2  # dimensionality of the domain
    omega = 2 * torch.pi  # Total solid angle for d = 2
    n = 2

    # Create indices array and norms
    I, J = S2.shape
    a = torch.repeat_interleave(torch.arange(I), I).view(-1, 1)
    b = torch.arange(J - 1, -1, -1).repeat(J).view(-1, 1)
    S2_indices = torch.cat((a, b), dim=1)
    S2_indices -= int(size / 2)
    norms = torch.norm(S2_indices.float(), dim=1)
    valid_mask = norms != 0
    S2_indices = S2_indices[valid_mask]


    (J, I, L, K) = S3.shape
    S3 = S3.permute(1, 0, 3, 2)  # Equivalent to NumPy transpose

    # Generate all combinations of indices for S3
    S3_indices = (torch.stack(torch.meshgrid(torch.arange(I), torch.arange(J), torch.arange(K), torch.arange(L)),
                           dim=-1) - int(size / 2)).reshape(-1, 4)

    # r1 and r2 based on indices
    r1 = S3_indices[:, :2].float()
    r2 = S3_indices[:, 2:].float()

    # Compute norms and create masks for nonzero norms

    valid_r1 = torch.norm(r1, dim=1) != 0
    valid_r2 = torch.norm(r2, dim=1) != 0
    valid_mask = valid_r1 & valid_r2

    S3_indices = r1[valid_mask]
    S3_indices_r2 = r2[valid_mask]


    # Input tensors
    S2_indices = S2_indices.to(device=device)  # Move to GPU if available
    S3_indices = S3_indices.to(device=device)  # Move to GPU if available
    S3_indices_r2 = S3_indices_r2.to(device=device)  # Move to GPU if available

    # ----------------------------------------------------------------------
    # Step 1: Create a mapping from S2_indices to their row indices.
    # ----------------------------------------------------------------------
    # Convert S2_indices to a contiguous tensor (optional for efficiency)
    S2_indices = S2_indices.contiguous()

    # Create a flat tensor for comparison, e.g., (x, y) -> 2D index
    # Flatten each 2D row to a unique integer using base conversion
    max_val = S2_indices.max().item() + 1
    S2_flat = S2_indices[:, 0] * max_val + S2_indices[:, 1]

    # Similarly, flatten S3_indices for comparison
    S3_flat = S3_indices[:, 0] * max_val + S3_indices[:, 1]
    S3_flat_r2 = S3_indices_r2[:, 0] * max_val + S3_indices_r2[:, 1]


    sorted_S2_flat, sort_indices = torch.sort(S2_flat)  # Sort S2_flat
    S3_sorted_indices = torch.searchsorted(sorted_S2_flat, S3_flat)
    S3_sorted_indices_r2 = torch.searchsorted(sorted_S2_flat, S3_flat_r2)

    valid_mask = (S3_sorted_indices >= 0) & (S3_sorted_indices < len(sorted_S2_flat))
    S3_matched = sort_indices[S3_sorted_indices[valid_mask]].to(device=device)

    valid_mask = (S3_sorted_indices_r2 >= 0) & (S3_sorted_indices_r2 < len(sorted_S2_flat))
    S3_matched_r2 = sort_indices[S3_sorted_indices_r2[valid_mask]].to(device=device)

    # ----------------------------------------------------------------------
    # Step 3: Retrieve corresponding T matrices
    # ----------------------------------------------------------------------
    # Use advanced indexing to gather T matrices
    T1 = new_T[S3_matched]
    T2 = new_T[S3_matched_r2]
    # Ensure final shape matches (16769025, 2, 2)
    return T1, T2


def split_into_patches(image, patch_size):
    patches = []
    h, w = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches


def get_patch_npcf(microstructure,threepcf = True):
    patches = split_into_patches(microstructure, 64)
    patches = torch.as_tensor(np.array(patches))
    phi = patches.float().mean()
    S2 = twopcf_batched(patches)
    S2 = S2.mean(dim=0)
    S3 = None
    if threepcf:
        S3 = threepcf_fullset_batched(patches)
        S3 = S3.mean(dim=0)

    return phi, S2, S3

def save_patches(patches, output_dir="patches", is_grayscale=True):
    """
    Save patches as image files in a specified directory.

    Args:
        patches: List of image patches (NumPy arrays)
        output_dir: Directory to save patches
        is_grayscale: True for grayscale images, False for RGB
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, patch in enumerate(patches):
        # Create filename with index
        filename = os.path.join(output_dir, f"patch_{idx:04d}.png")

        # For empty patches, skip saving
        if patch.size == 0:
            continue

        # Handle different image types
        if is_grayscale:
            plt.imsave(filename, patch, cmap='gray')
        else:
            plt.imsave(filename, patch)

        print(f"Saved {filename}")


def data_preprocessing(means, length_scales_x, length_scales_y, base_path='./microstructure_data'):
    """
    Load, concatenate, and randomize ground truth and microstructure data from .npz files.

    Args:
        means (list): List of mean values for data files.
        length_scales_x (list): List of x length scale values for data files.
        length_scales_y (list): List of y length scale values for data files.
        base_path (str): Base directory path containing the .npz files.

    Returns:
        tuple: PyTorch tensors for ground truth and microstructures, randomized.
    """
    # Initialize lists to store data
    gt_list = []
    microstructure_list = []
    delta_2_list = []
    delta_3_list = []
    S2_list = []
    S3_list = []

    # Loop through all combinations of mean, length_scale_x, and length_scale_y
    for mean in means:
        for length_scale_x in length_scales_x:
            for length_scale_y in length_scales_y:
                # Construct the file path
                filepath = f'{base_path}/mean_{mean}_x_{length_scale_x}_y_{length_scale_y}.npz'

                try:
                    # Load the data for the current combination
                    data = np.load(filepath)

                    # Append ground truth and microstructures to lists
                    gt_list.append(data['gt'])
                    microstructure_list.append(data['microstructures'])
                    delta_2_list.append(data['Delta_2_list'])
                    delta_3_list.append(data['Delta_3_list'])
                    S2_list.append(data['S2_list'])
                    S3_list.append(data['S3_list'])


                except (OSError, ValueError, zipfile.BadZipFile) as e:
                    print(f"File not found: {filepath}. Skipping.")
    return gt_list, microstructure_list, delta_2_list, delta_3_list, S2_list, S3_list


import os
import numpy as np
import zipfile


def data_preprocessing_masked(means, length_scales_x, length_scales_y,
                              base_path='./microstructure_data_masked'):
    """
    Load and return the 'masked' S3 and Delta_3 from .npz files,
    plus everything else (gt, microstructures, etc.).

    The .npz files here are created by 'cleanup_dataset.py' and
    contain 'S3_masked' and 'Delta_3_masked' rather than the full arrays.

    Returns:
        (gt_list, microstructure_list,
         delta_2_list, delta_3_masked_list,
         S2_list, S3_masked_list)
    """
    gt_list = []
    microstructure_list = []
    delta_2_list = []
    delta_3_list_masked = []
    S2_list = []
    S3_list_masked = []

    for mean in means:
        for length_scale_x in length_scales_x:
            for length_scale_y in length_scales_y:
                filepath = f'{base_path}/mean_{mean}_x_{length_scale_x}_y_{length_scale_y}.npz'
                if not os.path.isfile(filepath):
                    print(f"File not found: {filepath}. Skipping.")
                    continue
                try:
                    data = np.load(filepath)
                except (OSError, ValueError, zipfile.BadZipFile) as e:
                    print(f"Could not load {filepath}: {e}")
                    continue

                # These are the arrays we have in the new .npz
                if not all(k in data for k in ["gt", "microstructures", "Delta_2_list", "S2_list",
                                               "S3_masked", "Delta_3_masked"]):
                    print(f"Missing keys in {filepath}. Skipping.")
                    continue

                gt_list.append(data["gt"])
                microstructure_list.append(data["microstructures"])
                delta_2_list.append(data["Delta_2_list"])
                S2_list.append(data["S2_list"])
                # The masked versions
                delta_3_list_masked.append(data["Delta_3_masked"])
                S3_list_masked.append(data["S3_masked"])

    return (gt_list, microstructure_list,
            delta_2_list, delta_3_list_masked,
            S2_list, S3_list_masked)

def compute_thetas(xs, ys):
    """
    Given arrays (or lists) of x and y coordinates,
    return an array of angles theta in radians
    where theta = arctan2(y, x).
    """
    # Convert xs, ys to NumPy arrays if they aren't already
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    # Compute angles in radians for each (x, y)
    thetas = np.arctan2(y_arr, x_arr)

    return thetas


def get_mixed_partial_loss(K,x):
    # K_{11} and K_{12}
    K11 = K[:, 0, 0]  # shape (N,)
    K12 = K[:, 0, 1]  # shape (N,)
    K21 = K[:, 1, 0]
    K22 = K[:, 1, 1]

    # We want d/dx2 of K11 and d/dx1 of K12:

    # grad_K11 will be the gradient of K11.sum() w.r.t. x, shape: (N, 2)
    grad_K11 = torch.autograd.grad(
        outputs=K11.sum(),
        inputs=x,
        create_graph=True  # so we can backprop through this operation too
    )[0]

    grad_K21 = torch.autograd.grad(
        outputs=K21.sum(),
        inputs=x,
        create_graph=True  # so we can backprop through this operation too
    )[0]


    # grad_K12 will be the gradient of K12.sum() w.r.t. x, shape: (N, 2)
    grad_K12 = torch.autograd.grad(
        outputs=K12.sum(),
        inputs=x,
        create_graph=True
    )[0]

    grad_K22 = torch.autograd.grad(
        outputs=K22.sum(),
        inputs=x,
        create_graph=True  # so we can backprop through this operation too
    )[0]

    # partial_x2 K11 is the second component of grad_K11
    dK11_dx2 = grad_K11[:, 1]

    # partial_x1 K12 is the first component of grad_K12
    dK12_dx1 = grad_K12[:, 0]

    dK21_dx2 = grad_K21[:, 1]

    dK22_dx1 = grad_K22[:, 0]

    # The PDE residual (we want this to be zero)
    residual = dK11_dx2 - dK12_dx1

    residual_2 = dK21_dx2 - dK22_dx1

    # Define a PDE loss (e.g. MSE of the residual)
    mixed_partial_loss = (residual ** 2).mean()

    mixed_partial_loss_2 = (residual_2 ** 2).mean()

    return mixed_partial_loss + mixed_partial_loss_2


from scipy.optimize import curve_fit
def curve_fitting(T):
    # Simulated example data (replace with actual T_matrix values)
    theta = torch.linspace(-np.pi, np.pi, 100)  # Generate theta values

    # Assume T_matrix follows some known function (for example purposes)
    T_11 = 1 + 0.5 * torch.cos(2 * theta)  # Diagonal element (1,1)
    T_22 = 1 - 0.5 * torch.cos(2 * theta)  # Diagonal element (2,2)
    T_12 = 0.3 * torch.cos(2 * theta)  # Off-diagonal element (1,2) = (2,1)

    # Define the fitting function
    def cos2theta_fit(theta, A, B, C):
        return A + B * np.cos(2 * theta) + C * np.sin(2 * theta)

    # Fit the curve to the simulated data
    popt_11, _ = curve_fit(cos2theta_fit, theta.numpy(), T_11.numpy())
    popt_22, _ = curve_fit(cos2theta_fit, theta.numpy(), T_22.numpy())
    popt_12, _ = curve_fit(cos2theta_fit, theta.numpy(), T_12.numpy())

    # Generate fitted curves
    T_11_fit = cos2theta_fit(theta.numpy(), *popt_11)
    T_22_fit = cos2theta_fit(theta.numpy(), *popt_22)
    T_12_fit = cos2theta_fit(theta.numpy(), *popt_12)

    # Plot results
    plt.figure(figsize=(10, 6))

    plt.plot(theta.numpy(), T_11.numpy(), 'r.', label="T_11 (data)")
    plt.plot(theta.numpy(), T_11_fit, 'r-', label="T_11 (fit)")

    plt.plot(theta.numpy(), T_22.numpy(), 'b.', label="T_22 (data)")
    plt.plot(theta.numpy(), T_22_fit, 'b-', label="T_22 (fit)")

    plt.plot(theta.numpy(), T_12.numpy(), 'g.', label="T_12 (data)")
    plt.plot(theta.numpy(), T_12_fit, 'g-', label="T_12 (fit)")

    plt.plot(theta.numpy(), np.cos(2 * theta.numpy()), 'k--', label="cos(2θ)")

    plt.xlabel("Theta (radians)")
    plt.ylabel("T_matrix elements")
    plt.legend()
    plt.title("Curve Fitting for T_matrix Elements vs cos(2θ)")
    plt.show()


def fourier_regularization(model, penalty_scale=1e-3, weight_by_n=True):
    """
    Compute a regularization term that penalizes large Fourier coefficients
    in ang_a, ang_b, and ang_c.

    Args:
        model         : Your FourierExpansion2x2 instance.
        penalty_scale : Overall scale of the regularization term.
        weight_by_n   : If True, multiply by n^2 (or n) to penalize
                        high frequencies more strongly.

    Returns:
        reg_term: A scalar tensor with the regularization cost.
    """
    reg_term = 0.0

    # Each of ang_a, ang_b, ang_c has shape [2*N], which are the (A0,B0,A1,B1,...).
    # We'll view them as [N, 2], where index i corresponds to frequency i.
    for ang_params in [model.ang_a, model.ang_b, model.ang_c]:
        # shape: (N, 2)
        ang_params_2d = ang_params.view(-1, 2)

        # Optionally skip the n=0 term (DC component) if you don't want to penalize it:
        # Or simply include it too. Your choice.
        for n in range(model.N):
            # A_n and B_n
            A_n = ang_params_2d[n, 0]
            B_n = ang_params_2d[n, 1]

            # Basic L2 penalty on these coefficients:
            coeff_sq = A_n**2 + B_n**2

            if weight_by_n:
                # Penalize higher frequencies more strongly, e.g. by n^2:
                reg_term += (n**2) * coeff_sq
            else:
                # Uniform weight:
                reg_term += coeff_sq

    return penalty_scale * reg_term