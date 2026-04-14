import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, lsqr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.io import loadmat
import requests
import io
from skimage.transform import resize

def rescale_binary_matrix(matrix, target_dim):
    """
    Rescales a binary square matrix to a given target dimension.

    Parameters:
    - matrix (np.ndarray): The binary square matrix to be rescaled.
    - target_dim (int): The desired dimension of the output square matrix.

    Returns:
    - np.ndarray: The rescaled binary square matrix.
    """
    # Ensure input is a numpy array
    matrix = np.asarray(matrix)


    # Use skimage's resize with 'nearest' mode to preserve binary values
    resized_matrix = resize(matrix, (target_dim, target_dim), order=0, preserve_range=True, anti_aliasing=False)

    # Convert back to binary
    resized_matrix = (resized_matrix > 0.5).astype(int)

    return resized_matrix

def generate_correlated_random_field(size, mean, length_scale_x, length_scale_y, seed=None):
    """
    Generates a square 2D spatially correlated random field and binarizes it.

    Parameters:
    - size (int): The dimensions of the square random field (size x size).
    - mean (float): The mean value of the random field.
    - cov (float): The variance of the Gaussian random field.
    - correlation_length (float): The correlation length, controlling spatial correlation.
    - threshold (float, optional): The threshold for binarization.
                                   If None, the mean of the generated field will be used.

    Returns:
    - binary_field (np.ndarray): The binarized 2D random field.
    - random_field (np.ndarray): The generated random field before binarization.
    """
    nx = size
    ny = size
    if seed != None:
      np.random.seed(seed=seed)
    # Generate a grid of frequencies
    kx = np.fft.fftfreq(nx, 1. / nx)
    ky = np.fft.fftfreq(ny, 1. / ny)
    kx, ky = np.meshgrid(kx, ky)

    # Compute the power spectrum with different scaling for x and y dimensions
    power_spectrum = np.exp(-0.5 * ((kx * length_scale_x) ** 2 + (ky * length_scale_y) ** 2))

    # Generate a grid of complex Gaussian random numbers
    random_complex = np.random.normal(0, 1, (nx, ny)) + 1j * np.random.normal(0, 1, (nx, ny))

    # Compute the inverse Fourier transform to obtain the random field
    random_field = np.real(np.fft.ifft2(np.fft.fft2(random_complex) * np.sqrt(power_spectrum)))

    # Normalize the field to have the desired mean and variance
    random_field = random_field - np.mean(random_field)
    random_field = random_field / np.std(random_field)
    # random_field = mean + random_field * np.sqrt(cov)
    random_field = random_field + mean
    flat_vals = random_field.ravel()
    T = np.quantile(flat_vals, 1 - mean)
    # Set the threshold for binarization
    # if threshold is None:
    #     threshold = np.mean(random_field)

    # Binarize the random field
    binary_field = (random_field > T).astype(float)

    return binary_field, random_field
#
# import numpy as np
#
# def generate_correlated_random_field(
#     size,
#     vol_frac,
#     length_scale_major,
#     length_scale_minor,
#     angle_deg=0.0,
#     seed=None
# ):
#     """
#     Generates a square 2D *rotated* anisotropic Gaussian random field and a binarized phase map.
#
#     Parameters
#     ----------
#     size : int
#         Dimension of the square field (size x size).
#     vol_frac : float in (0,1)
#         Target volume fraction of the '1' phase after binarization.
#     length_scale_major : float
#         Correlation length along the *major* principal direction.
#     length_scale_minor : float
#         Correlation length along the *minor* principal direction.
#     angle_deg : float, default 30
#         Rotation angle (degrees) of the principal correlation directions *relative to x-axis*.
#         Any non-multiple of 90° will break orthotropy in the x–y basis.
#     seed : int or None
#         RNG seed.
#
#     Returns
#     -------
#     binary_field : (size, size) float array
#         Binarized field (0/1) with ~vol_frac of 1s.
#     random_field : (size, size) float array
#         Zero-mean, unit-std Gaussian field (before shifting to achieve vol_frac).
#     """
#     nx = ny = size
#     if seed is not None:
#         np.random.seed(seed)
#
#     # Frequencies on a periodic grid (FFT assumes periodic BCs)
#     kx = np.fft.fftfreq(nx, d=1.0)  # cycles per grid length
#     ky = np.fft.fftfreq(ny, d=1.0)
#     kx, ky = np.meshgrid(kx, ky, indexing='xy')
#
#     # Rotate the wavevectors by angle θ
#     th = np.deg2rad(angle_deg)
#     kxr =  kx*np.cos(th) + ky*np.sin(th)
#     kyr = -kx*np.sin(th) + ky*np.cos(th)
#
#     # Anisotropic Gaussian power spectrum, principal axes aligned with (kxr, kyr)
#     # Note: larger length_scale => tighter falloff in k-space
#     # Add tiny epsilon at k=0 to avoid divide-by-zero / NaNs
#     eps = 1e-12
#     ps = np.exp(-0.5 * ((kxr * length_scale_major)**2 + (kyr * length_scale_minor)**2)) + eps
#
#     # White noise -> filter in Fourier domain -> correlated field
#     white = np.random.normal(0.0, 1.0, (ny, nx))
#     Fw = np.fft.fft2(white)
#     # Multiply by sqrt of power spectrum (amplitude filter), then inverse FFT
#     Fr = Fw * np.sqrt(ps)
#     field = np.fft.ifft2(Fr).real
#
#     # Normalize to zero-mean, unit-std
#     field -= field.mean()
#     std = field.std()
#     if std > 0:
#         field /= std
#
#     # Binarize by quantile to hit target volume fraction
#     T = np.quantile(field, 1.0 - vol_frac)
#     binary = (field > T).astype(float)
#
#     return binary, field


def generate_microstructure_from_url(mat, id, size):
    """
    Reads microstructure data from a .mat file at a given URL and resizes it.

    Parameters:
    - mat (str): The URL of the .mat file.
    - id (int): Index of the data point to read.
    - size (int): The desired output size for the microstructure.

    Returns:
    - microstructure (np.ndarray): The resized microstructure.
    """
    # Download the file
    response = requests.get(mat)
    response.raise_for_status()  # Ensure the request was successful

    # Load the .mat file
    mat_data = loadmat(io.BytesIO(response.content))

    # Extract and resize the microstructure
    microstructure = mat_data['Data']
    microstructure = microstructure[id, :].reshape([128, 128])  # take the first data point
    microstructure = rescale_binary_matrix(microstructure, size)

    return microstructure

def get_microstructure(mode='url', mat_url=None, id=0, size=512, mean=0.5, length_scale_x=0.5, length_scale_y=0.5, threshold=None,seed=None, binary=True):
    """
    Generates or reads a microstructure based on the mode specified.

    Parameters:
    - mode (str): 'url' to read from a URL or 'generate' to create a random field.
    - mat_url (str): The URL of the .mat file (required if mode='url').
    - id (int): Index of the data point to read (if mode='url').
    - size (int): The desired output size for the microstructure.
    - mean (float): The mean value of the random field (if mode='generate').
    - cov (float): The variance for the Gaussian random field (if mode='generate').
    - correlation_length (float): The correlation length for the random field (if mode='generate').
    - threshold (float, optional): The threshold for binarization (if mode='generate').

    Returns:
    - microstructure (np.ndarray): The generated or read microstructure.
    """
    if mode == 'url':
        if mat_url is None:
            raise ValueError("URL must be provided for mode='url'.")
        return generate_microstructure_from_url(mat_url, id, size)
    elif mode == 'generate':
        binary_field, random_field = generate_correlated_random_field(size, mean, length_scale_x, length_scale_y, seed)
        return binary_field
    else:
        raise ValueError("Invalid mode. Choose 'url' or 'generate'.")

# # Example usage:

# # To load microstructure from a URL:
# mat_url = "https://github.com/shengcheng/Material_Reconstruction_Correlation/raw/505fd6052acf51dc8dc6bb5744823244343edec0/sandstone_data.mat"
# microstructure_from_url = get_microstructure(mode='url', mat_url=mat_url, id=0, size=512)

# # To generate a spatially correlated random microstructure:
# random_microstructure = get_microstructure(mode='generate', size=512, mean=0.5, cov=0.1, correlation_length=5)

# print("Microstructure from URL:")
# print(microstructure_from_url)
# print("\nRandomly Generated Spatially Correlated Microstructure:")
# print(random_microstructure)
if __name__ == '__main__':
    gt_size = 64
    mean = 0.5
    length_scale_x = 0.5
    length_scale_y = 0.5
    threshold = None
    seed = 2004
    microstructure_gt = get_microstructure(mode='generate', size=gt_size, mean=mean, length_scale_x=length_scale_x,
                                               length_scale_y=length_scale_y, threshold=threshold, seed=seed,
                                               binary=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(microstructure_gt,
               cmap='gray',  # or cmap='binary'
               interpolation='nearest',
               origin='lower',  # optional: sets (0,0) in the lower‐left
               aspect='equal')  # each pixel is square
    plt.axis('off')  # turn off axes if you like
    plt.title('2D Binary Microstructure')
    plt.show()