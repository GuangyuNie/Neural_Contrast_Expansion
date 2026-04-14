import torch

def twopcf(image_tensor):
    assert image_tensor.ndim == 2, "Input must be a 2D image."

    # Perform Fourier transform using PyTorch (with GPU if tensor is on CUDA)
    F_k = torch.fft.fft2(image_tensor)

    # Compute the power spectrum
    power_spectrum = F_k * torch.conj(F_k)

    # Compute the inverse Fourier transform of the power spectrum
    correlations = torch.fft.ifft2(power_spectrum).real

    # Shift the zero-frequency component to the center of the spectrum
    correlations = torch.fft.fftshift(correlations)

    # Normalize by the number of elements in the image
    return correlations / image_tensor.numel()


def threepcf_fullset(image_tensor):
    """Consider the field have n elements in total, the complexity of this algorithm is O(nlogn)"""
    assert image_tensor.ndim == 2
    F_k = torch.fft.fft2(image_tensor)

    # Get the dimensions of the array
    n, m = F_k.shape

    # Step 2: Create grid of indices using torch.meshgrid
    k1, l1, k2, l2 = torch.meshgrid(torch.arange(n, device=image_tensor.device),
                                    torch.arange(m, device=image_tensor.device),
                                    torch.arange(n, device=image_tensor.device),
                                    torch.arange(m, device=image_tensor.device), indexing='ij')

    # Step 3: Compute the indices for the bispectrum
    k3 = (k1 + k2) % n
    l3 = (l1 + l2) % m

    # Step 4: Calculate the bispectrum using broadcasting
    bispectrum = F_k[k1, l1] * F_k[k2, l2] * torch.conj(F_k[k3, l3])

    correlations = torch.fft.ifftn(bispectrum,dim = (-3,-2,-1,0)).real
    correlations = torch.fft.fftshift(correlations)
    return correlations / image_tensor.numel()
import torch


def twopcf_batched(image_tensor):
    """
    Batched 2-point correlation function (autocorrelation)
    Input shape: (batch, height, width)
    Output shape: (batch, height, width)
    """
    assert image_tensor.ndim == 3, "Input must be a 3D tensor (batch, height, width)"

    # FFT over spatial dimensions (last two dims)
    F_k = torch.fft.fft2(image_tensor)

    # Power spectrum calculation
    power_spectrum = F_k * torch.conj(F_k)

    # Inverse FFT and shift
    correlations = torch.fft.ifft2(power_spectrum).real
    correlations = torch.fft.fftshift(correlations, dim=(1,2))

    # Normalize by number of pixels
    return correlations / (image_tensor.shape[-2] * image_tensor.shape[-1])


def threepcf_fullset_batched(image_tensor):
    """
    Batched 3-point correlation function (bispectrum)
    Input shape: (batch, height, width)
    Output shape: (batch, height, width, height, width)
    """
    assert image_tensor.ndim == 3, "Input must be a 3D tensor (batch, height, width)"
    B, n, m = image_tensor.shape
    device = image_tensor.device

    # FFT over spatial dimensions
    F_k = torch.fft.fft2(image_tensor)  # (B, n, m)

    # Create index grids (precompute once per image size)
    k1, l1, k2, l2 = torch.meshgrid(
        torch.arange(n, device=device),
        torch.arange(m, device=device),
        torch.arange(n, device=device),
        torch.arange(m, device=device),
        indexing='ij'
    )

    # Calculate conjugate indices using modulo arithmetic
    k3 = (k1 + k2) % n
    l3 = (l1 + l2) % m

    # Gather values using advanced indexing (memory intensive!)
    F_k1 = F_k[:, k1, l1]  # (B, n, m, n, m)
    F_k2 = F_k[:, k2, l2]  # (B, n, m, n, m)
    F_k3_conj = torch.conj(F_k[:, k3, l3])  # (B, n, m, n, m)

    # Calculate bispectrum
    bispectrum = F_k1 * F_k2 * F_k3_conj

    # 4D inverse FFT over spatial dimensions
    correlations = torch.fft.ifftn(bispectrum, dim=(1, 2, 3, 4)).real

    # Shift all spatial dimensions
    correlations = torch.fft.fftshift(correlations, dim=(1, 2, 3, 4))

    # Normalize by number of pixels
    return correlations / (n * m)

def fourpcf_fullset(image_tensor):
    """Consider the field have n elements in total, the complexity of this algorithm is O(nlogn)"""
    assert image_tensor.ndim == 2
    F_k = torch.fft.fft2(image_tensor)

    # Get the dimensions of the array
    n, m = F_k.shape

    # Step 2: Create grid of indices using torch.meshgrid
    k1, l1, k2, l2, k3, l3 = torch.meshgrid(torch.arange(n, device=image_tensor.device),
                                            torch.arange(m, device=image_tensor.device),
                                            torch.arange(n, device=image_tensor.device),
                                            torch.arange(m, device=image_tensor.device),
                                            torch.arange(n, device=image_tensor.device),
                                            torch.arange(m, device=image_tensor.device), indexing='ij')

    # Step 3: Compute the indices for the trispectrum
    k4 = (k1 + k2 + k3) % n
    l4 = (l1 + l2 + l3) % m

    # Step 4: Calculate the trispectrum using broadcasting
    trispectrum = F_k[k1, l1] * F_k[k2, l2] * F_k[k3, l3] * torch.conj(F_k[k4, l4])

    correlations = torch.fft.ifftn(trispectrum).real
    correlations = torch.fft.fftshift(correlations)
    return correlations / image_tensor.numel()


def threepcf_on_demand(image_tensor, x1, y1, x2, y2):
    """Instead of getting a whole NPCF matrix,
    this function directly outputs a single 3PCF scalar given any configuration (x1, y1, x2, y2).
    This is useful when you don't want to calculate the whole matrix,
    but only want to get a value given certain configurations."""
    assert image_tensor.ndim == 2
    f1 = image_tensor
    f2 = torch.roll(image_tensor, shifts=(x1, y1), dims=(0, 1))
    f3 = torch.roll(image_tensor, shifts=(x2, y2), dims=(0, 1))

    P = f1 * f2 * f3
    return torch.mean(P)
