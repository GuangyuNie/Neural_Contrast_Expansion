import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from helper.utils import *
from sce_pipeline import *
from model.fourier_conductivity import *
from model.bessel_fourier_wave import *


def coeff_l1_loss(model):
    """
    Paper: ||C||_1  (preferably only on coefficient-like params).
    If no obvious coefficient params exist, fall back to all params.
    """
    coeff_keys = ["c_", "coef", "coeff", "cossin", "fourier", "bessel"]
    named = list(model.named_parameters())
    picked = [(n, p) for (n, p) in named if any(k in n.lower() for k in coeff_keys)]
    if len(picked) == 0:
        return sum(p.abs().sum() for p in model.parameters())
    return sum(p.abs().sum() for _, p in picked)


def laplacian(u, x):
    """
    u: [M,1] scalar field u(x)
    x: [M,2] coords, requires_grad=True
    return: [M,1] Δu
    """
    grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]  # [M,2]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy


def physics_pde_loss_G(model, coords, eps=0.02, use_delta=False):
    """
    Paper: || L[G] - delta ||^2
    For conduction Laplace: L = Δ (away from origin, ΔG=0).
    Requires model.green(coords)->[M,1] or model.forward_G(coords)->[M,1].
    """
    coords = coords.requires_grad_(True)

    if hasattr(model, "green"):
        G = model.green(coords)
    elif hasattr(model, "forward_G"):
        G = model.forward_G(coords)
    else:
        raise RuntimeError(
            "Paper PDE loss needs scalar Green G. Please add model.green(coords)->[M,1] "
            "(recommended), or set lambda_PDE=0."
        )

    if G.ndim == 1:
        G = G[:, None]

    LG = laplacian(G, coords)

    if not use_delta:
        r = torch.sqrt((coords ** 2).sum(dim=1, keepdim=True) + 1e-12)
        mask = (r > eps).float()
        return ((LG * mask) ** 2).mean()

    # smooth delta (2D Gaussian)
    r2 = (coords ** 2).sum(dim=1, keepdim=True)
    delta_eps = torch.exp(-r2 / (eps ** 2)) / (torch.pi * eps ** 2)
    return ((LG - delta_eps) ** 2).mean()


def hessian_integrability_loss_H(model, coords):
    """
    Paper (Hessian-case only): integrability/curl-free constraint.
    For each i: curl( (H_{i1},H_{i2}) ) = ∂x H_{i2} - ∂y H_{i1} = 0.
    Requires model.hessian(coords)->[M,2,2] or model(coords)->[M,2,2].
    """
    coords = coords.requires_grad_(True)

    if hasattr(model, "hessian"):
        H = model.hessian(coords)
    else:
        out = model(coords)
        H = out if (isinstance(out, torch.Tensor) and out.ndim == 3 and out.shape[-2:] == (2, 2)) else None

    if H is None:
        raise RuntimeError(
            "Hessian integrability loss needs H(coords)->[M,2,2]. "
            "Please implement model.hessian(coords) or make model(coords) return [M,2,2]."
        )

    loss = 0.0
    for i in (0, 1):
        Hix = H[:, i, 0:1]  # [M,1]
        Hiy = H[:, i, 1:2]  # [M,1]

        dHiy = torch.autograd.grad(Hiy.sum(), coords, create_graph=True)[0]  # [M,2]
        dHix = torch.autograd.grad(Hix.sum(), coords, create_graph=True)[0]  # [M,2]

        dHiy_dx = dHiy[:, 0:1]
        dHix_dy = dHix[:, 1:2]
        curl_i = dHiy_dx - dHix_dy
        loss = loss + (curl_i ** 2).mean()

    return loss


# =========================
# Main training
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0, help='Which GPU device to use (default: 0)')
    parser.add_argument('--version', type=int, default=0, help='version of the training code used (default: 0)')
    parser.add_argument('--model_id', type=str, default='bessel',
                        choices=['toy', 'siren', 'fourier', 'bessel'],
                        help='Which model to train (default: "bessel")')
    parser.add_argument('--do_3pcf', action='store_true',
                        help='Enable 3pcf (default: False). If not set, uses 2pcf.')
    parser.add_argument('--save_dir', type=str, default='./kernel_training_result_paper',
                        help='Directory to save checkpoints')
    args = parser.parse_args()

    config = load_config('config.yaml')
    sigma0 = config['sigma0']
    sigma1 = config['sigma1']
    size = config['size']
    d = config['d']
    gt_size = config['gt_size']
    mean_cfg = config.get('mean', None)
    seed = config.get('seed', 0)

    device = torch.device(f"cuda:{args.device_id}")
    torch.manual_seed(seed)

    # ======= data =======
    x_scales = [0.001]
    y_scales = [0.01]
    means = [0.7] if mean_cfg is None else [mean_cfg]

    gt_list, microstructure_list, delta_2_list, delta_3_list, S2_list, S3_list = data_preprocessing_masked(
        means, x_scales, y_scales, base_path='microstructure_data'
    )

    gt_tensor = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in gt_list])
    microstructure_tensor = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in microstructure_list])
    delta_2_tensor = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in delta_2_list])
    delta_3_tensor = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in delta_3_list])
    S2_tensor = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in S2_list])
    S3_tensor = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in S3_list])

    mask = torch.load('./helper/mask.pt')

    microstructure_tensor = microstructure_tensor.view(-1, size, size)
    gt_tensor = gt_tensor.view(-1, 4)
    delta_2_tensor = delta_2_tensor.view(-1, size ** 2 - 1)
    delta_3_tensor = delta_3_tensor.view(-1, 28649)
    S2_tensor = S2_tensor.view(-1, size, size)
    S3_tensor = S3_tensor.view(-1, 28649)

    perm = torch.randperm(len(gt_tensor))
    microstructure_tensor = microstructure_tensor[perm]
    gt_tensor = gt_tensor[perm]
    delta_2_tensor = delta_2_tensor[perm]
    delta_3_tensor = delta_3_tensor[perm]
    S2_tensor = S2_tensor[perm]
    S3_tensor = S3_tensor[perm]

    class CompositeDataset(Dataset):
        def __init__(self, microstructure_tensor, gt_tensor, delta_2_tensor, delta_3_tensor, S2_tensor, S3_tensor):
            self.microstructure = microstructure_tensor.unsqueeze(1).float()  # [N,1,H,W]
            self.targets = gt_tensor.float()  # [N,4]
            self.delta_2_tensor = delta_2_tensor
            self.delta_3_tensor = delta_3_tensor
            self.S2_tensor = S2_tensor
            self.S3_tensor = S3_tensor

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return (self.microstructure[idx],
                    self.delta_2_tensor[idx],
                    self.delta_3_tensor[idx],
                    self.S2_tensor[idx],
                    self.S3_tensor[idx]), self.targets[idx]

    dataset = CompositeDataset(microstructure_tensor, gt_tensor, delta_2_tensor, delta_3_tensor, S2_tensor, S3_tensor)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ======= pipeline / model =======
    loss_fn = nn.MSELoss()

    beta = (sigma1 - sigma0) / (sigma1 + (d - 1) * sigma0)
    do_3pcf = args.do_3pcf
    n_npcf = 3 if do_3pcf else 2
    sce_pipeline = EffectiveConductivityNPCF(size, sigma0, sigma1, n_npcf, d, device, gt_size)

    # Use your existing model. Paper constraints need:
    # - model(coords)->[M,2,2] or model.hessian(coords)->[M,2,2]
    # - model.green(coords)->[M,1] for PDE regularization
    if args.model_id == "bessel":
        model = BesselFourier2x2().to(device)
    else:
        # fallback (use your own choices)
        model = BesselFourier2x2().to(device)

    # ======= optimizer =======
    from torchmin import Minimizer
    optimizer = Minimizer(model.parameters(), method="trust-ncg")

    # ======= loaders =======
    batch_size = 48
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    lambda_C = 1e-4     # ||C||_1
    lambda_PDE = 1e-2   # ||L[G]-delta||^2
    lambda_INT = 1e-2   # integrability (Hessian-case only)
    lambda_extra = 0.0  # optional extra (leave 0 to match paper more strictly)

    # Conduction (randomness in 2nd-order term) -> Hessian-kernel case
    is_hessian_case = True

    # ======= training =======
    num_epochs = 20000
    train_losses, val_losses = [], []

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    early_stop_threshold = 0.01

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0.0

        for (microstructure, delta_2, delta_3, S2, S3), targets in train_loader:
            microstructure = microstructure.to(device)
            S2 = S2.to(device)
            S3 = S3.to(device)
            targets = targets.to(device)

            def closure():
                optimizer.zero_grad(set_to_none=True)

                batch_loss = 0.0
                B = microstructure.size(0)

                # sample coords once per batch for physics losses (cheaper & closer to paper)
                coords = (torch.rand(256, 2, device=device) - 0.5)

                # physics losses (shared across samples in batch)
                # If your model doesn't have green(), either add it or set lambda_PDE=0.
                pde_loss = physics_pde_loss_G(model, coords, eps=0.02, use_delta=False) if lambda_PDE > 0 else 0.0
                int_loss = hessian_integrability_loss_H(model, coords) if (is_hessian_case and lambda_INT > 0) else 0.0
                l1C = coeff_l1_loss(model) if lambda_C > 0 else 0.0

                for i in range(B):
                    phi = torch.mean(microstructure[i][0]).item()
                    S2_ = S2[i]
                    S3_ = S3[i]

                    # NN kernel -> A2/A3 -> D
                    if do_3pcf:
                        A2 = sce_pipeline.compute_A2(beta, phi, S2_, use_NN=True, model=model)
                        A3 = sce_pipeline.compute_A3(beta, phi, S2_, S3_, use_NN=True, model=model, mask=mask)
                        D = sce_pipeline.compute_D(A2=A2, A3=A3, phi=phi)
                    else:
                        A2 = sce_pipeline.compute_A2(beta, phi, S2_, use_NN=True, model=model)
                        D = sce_pipeline.compute_D(A2=A2, A3=None, phi=phi)

                    # Paper supervision: D_target from gt Σ_e
                    D_target = sce_pipeline.compute_D_target(gt=targets[i].reshape(2, 2), phi=phi, beta=beta)

                    # Data term in D-space (you can keep anisotropic weighting)
                    data_loss = (
                        10.0 * loss_fn(D[0, 0], D_target[0, 0]) +
                        10.0 * loss_fn(D[1, 1], D_target[1, 1]) +
                        1.0 * loss_fn(D[0, 1], D_target[0, 1]) +
                        1.0 * loss_fn(D[1, 0], D_target[1, 0])
                    )

                    loss_i = data_loss
                    batch_loss = batch_loss + loss_i

                batch_loss = batch_loss / B

                # Add paper regularizers (once per batch, consistent with paper "global" Θ)
                batch_loss = batch_loss + lambda_C * l1C + lambda_PDE * pde_loss + lambda_INT * int_loss

                # Optional extras (only if your model implements them; keep 0 to match paper)
                if lambda_extra > 0:
                    if hasattr(model, "l2norm_loss"):
                        batch_loss = batch_loss + lambda_extra * model.l2norm_loss(coords)
                    if hasattr(model, "lasso_individual"):
                        batch_loss = batch_loss + lambda_extra * model.lasso_individual()

                return batch_loss

            loss = optimizer.step(closure)
            total_train_loss += loss.item()

        # ======= validation (use Σ space like your original, but could also validate in D-space) =======
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for (microstructure, delta_2, delta_3, S2, S3), targets in val_loader:
                microstructure = microstructure.to(device)
                S2 = S2.to(device)
                S3 = S3.to(device)
                targets = targets.to(device)

                batch_val_loss = 0.0
                B = microstructure.size(0)
                for i in range(B):
                    phi = torch.mean(microstructure[i][0]).item()
                    S2_ = S2[i]
                    S3_ = S3[i]

                    if do_3pcf:
                        A2 = sce_pipeline.compute_A2(beta, phi, S2_, use_NN=True, model=model)
                        A3 = sce_pipeline.compute_A3(beta, phi, S2_, S3_, use_NN=True, model=model, mask=mask)
                        D = sce_pipeline.compute_D(A2=A2, A3=A3, phi=phi)
                    else:
                        A2 = sce_pipeline.compute_A2(beta, phi, S2_, use_NN=True, model=model)
                        D = sce_pipeline.compute_D(A2=A2, A3=None, phi=phi)

                    Sigma = sce_pipeline.compute_Sigma(D, phi=phi)  # [2,2]
                    tgt = targets[i].view(2, 2)

                    batch_val_loss = batch_val_loss + (
                        loss_fn(Sigma[0, 0], tgt[0, 0]) +
                        loss_fn(Sigma[1, 1], tgt[1, 1]) +
                        0.0 * loss_fn(Sigma[0, 1], tgt[0, 1]) +
                        0.0 * loss_fn(Sigma[1, 0], tgt[1, 0])
                    )

                total_val_loss += (batch_val_loss.item() / B)

        epoch_train_loss = total_train_loss / max(1, len(train_loader))
        epoch_val_loss = total_val_loss / max(1, len(val_loader))
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # Save checkpoint
        tag = "3pcf" if do_3pcf else "2pcf"
        torch.save(model, save_dir / f'kernel_learning_{tag}_{args.model_id}_v{args.version}_{epoch}.pt')

        if epoch_val_loss < early_stop_threshold:
            print(f"Validation loss {epoch_val_loss:.6f} < {early_stop_threshold}, early stop.")
            break


if __name__ == '__main__':
    main()
