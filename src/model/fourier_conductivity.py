import torch
import torch.nn as nn


class FourierExpansion2x2(nn.Module):

    def __init__(self, N=6, hidden=16, factor_out_r2=True):
        """
        Args:
            N            : number of Fourier modes for the angular part
            hidden       : hidden units in the radial MLP
            factor_out_r2: if True, multiply final by 1/r^2
        """
        super().__init__()
        torch.manual_seed(1996)
        self.N = N
        self.hidden = hidden
        self.factor_out_r2 = factor_out_r2

        # --- (1) Define learnable scale parameters c_a, c_b, c_c
        #         We'll exponentiate these at forward pass => alpha = exp(c)
        self.c_a = nn.Parameter(torch.zeros(1))
        self.c_b = nn.Parameter(torch.zeros(1))
        self.c_c = nn.Parameter(torch.zeros(1))

        # --- (2) Small radial MLPs: each outputs g(r), a single scalar
        def make_radial_mlp():
            return nn.Sequential(
                nn.Linear(1, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1)  # single scalar output
            )

        self.radialNetA = make_radial_mlp()
        self.radialNetB = make_radial_mlp()
        self.radialNetC = make_radial_mlp()

        # shape: (2*N,) for each entry
        self.ang_a = nn.Parameter(torch.zeros(2 * N))
        self.ang_b = nn.Parameter(torch.zeros(2 * N))
        self.ang_c = nn.Parameter(torch.zeros(2 * N))

        nn.init.normal_(self.ang_a, mean=0.0, std=.1)
        nn.init.normal_(self.ang_b, mean=0.0, std=.1)
        nn.init.normal_(self.ang_c, mean=0.0, std=.1)

    def forward(self, coords):
        """
        coords: Tensor of shape [batch_size, 2], i.e. (x, y).
        Returns: Tensor of shape [batch_size, 2, 2].
        """
        x = coords[:, 0]*(1024/64)
        y = coords[:, 1]*(1024/64)

        # radius + angle
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-10)
        theta = torch.atan2(y, x)
        alpha_a = torch.exp(self.c_a)  # scalar

        n_vals = 2*torch.arange(self.N, device=theta.device, dtype=theta.dtype).unsqueeze(0)
        cosn = torch.cos(theta.unsqueeze(-1) * n_vals)  # [bs, N]
        sinn = torch.sin(theta.unsqueeze(-1) * n_vals)  # [bs, N]

        def fourier_sum(ang_params, cosn, sinn):
            A_k = ang_params.view(-1, 2)[:, 0]  # shape [N]
            B_k = ang_params.view(-1, 2)[:, 1]  # shape [N]

            f_val = (cosn * A_k) + (sinn * B_k)  # shape [bs, N]
            return f_val.sum(dim=1)  # shape [bs]

        Atheta_a = fourier_sum(self.ang_a, cosn, sinn)  # [bs]
        Atheta_b = fourier_sum(self.ang_b, cosn, sinn)
        Atheta_c = fourier_sum(self.ang_c, cosn, sinn)

        a_val = 1.0 / (r ** (2 * alpha_a)) * Atheta_a
        b_val = 1.0 / (r ** (2 * alpha_a)) * Atheta_b
        c_val = 1.0 / (r ** (2 * alpha_a)) * Atheta_c

        # (E) Construct the final [batch_size, 2, 2] output
        # You said you want something like [[a, b], [b, c]]
        out = torch.stack([
            torch.stack([a_val, b_val], dim=1),
            torch.stack([b_val, c_val], dim=1)
        ], dim=1)  # shape [bs, 2, 2]

        return out

