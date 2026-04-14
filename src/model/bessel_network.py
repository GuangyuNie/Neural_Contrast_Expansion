# import torch
# import torch.nn as nn
# import math
# #
# # class BesselFourier2x2(nn.Module):
# #     """
# #     2×2 kernel parameterized by a Bessel–Fourier expansion with learnable power-law envelope:
# #
# #       K(r,θ) = r^{-α} [[a, b], [b, c]]
# #
# #     Each entry (a, b, c) is:
# #       \sum_{n=0}^N \sum_{m=1}^M [c_cos[n,m] J_n(α_nm r) cos(nθ)
# #                                     + c_sin[n,m] J_n(α_nm r) sin(nθ)],
# #     approximating J_n via its integral definition and then applying r^{-α}.
# #     α (envelope exponent) is now a learnable parameter initialized to 1.
# #     """
# #     def __init__(self, N: int, M: int, Q: int = 32):
# #         super().__init__()
# #         self.N = N
# #         self.M = M
# #         # learnable envelope exponent, init to 1.0
# #         self.alpha_env = nn.Parameter(torch.tensor(1.0))
# #         # learnable radial frequencies α_{n,m}
# #         self.alpha = nn.Parameter(torch.randn(N+1, M))
# #         # learnable Fourier coefficients for a, b, c
# #         self.c_a_cos = nn.Parameter(torch.zeros(N+1, M))
# #         self.c_a_sin = nn.Parameter(torch.zeros(N+1, M))
# #         self.c_b_cos = nn.Parameter(torch.zeros(N+1, M))
# #         self.c_b_sin = nn.Parameter(torch.zeros(N+1, M))
# #         self.c_c_cos = nn.Parameter(torch.zeros(N+1, M))
# #         self.c_c_sin = nn.Parameter(torch.zeros(N+1, M))
# #         # quadrature nodes for J_n integral approx
# #         phi = torch.linspace(0, math.pi, Q)
# #         self.register_buffer('phi', phi)
# #         self.register_buffer('sin_phi', torch.sin(phi))
# #         self.register_buffer('n_idx', torch.arange(N+1, dtype=torch.float32))
# #
# #     def forward(self, coords: torch.Tensor) -> torch.Tensor:
# #         # coords: [batch,2] -> (x,y)
# #         x, y = coords.unbind(-1)
# #         r = torch.sqrt(x**2 + y**2 + 1e-8)
# #         theta = torch.atan2(y, x)
# #
# #         # angular bases: cos(nθ), sin(nθ)
# #         n_idx = self.n_idx  # [N+1]
# #         cosn = torch.cos(theta.unsqueeze(-1) * n_idx)
# #         sinn = torch.sin(theta.unsqueeze(-1) * n_idx)
# #
# #         # compute J_n(alpha[n,m] * r) via integral definition
# #         arg = self.alpha.unsqueeze(0) * r.unsqueeze(1).unsqueeze(2)  # [batch, N+1, M]
# #         sin_part = arg.unsqueeze(-1) * self.sin_phi  # [batch, N+1, M, Q]
# #         n_part = n_idx.view(1, -1, 1, 1) * self.phi      # [1, N+1, 1, Q]
# #         J = torch.cos(sin_part - n_part).mean(dim=-1)   # [batch, N+1, M]
# #
# #         # combine radial J and angular coeffs for a, b, c
# #         coeffs_a = self.c_a_cos.unsqueeze(0) * cosn.unsqueeze(-1) + self.c_a_sin.unsqueeze(0) * sinn.unsqueeze(-1)
# #         coeffs_b = self.c_b_cos.unsqueeze(0) * cosn.unsqueeze(-1) + self.c_b_sin.unsqueeze(0) * sinn.unsqueeze(-1)
# #         coeffs_c = self.c_c_cos.unsqueeze(0) * cosn.unsqueeze(-1) + self.c_c_sin.unsqueeze(0) * sinn.unsqueeze(-1)
# #
# #         # sum over n,m -> [batch]
# #         a_val = (J * coeffs_a).sum(dim=(1,2))
# #         b_val = (J * coeffs_b).sum(dim=(1,2))
# #         c_val = (J * coeffs_c).sum(dim=(1,2))
# #
# #         # apply learnable power-law envelope r^{-alpha_env}
# #         envelope = r.pow(-self.alpha_env).unsqueeze(-1).unsqueeze(-1)  # [batch,1,1]
# #         # assemble 2×2 matrices and apply envelope
# #         out = torch.stack([
# #             torch.stack([a_val, b_val], dim=1),
# #             torch.stack([b_val, c_val], dim=1)
# #         ], dim=1) * envelope
# #
# #         return out
#
#
#
#
# class BesselFourier2x2(nn.Module):
#     """
#     V2: 2x2 Green's tensor with preserved physical interpretability (H0, H1, H2).
#     Implements optimizations from Strategy 1 and 3:
#     1.  [S1] Unified complex coefficient structure for all learnable branches.
#     2.  [S1] Unified internal learning function `_learn_component` to reduce code redundancy.
#     3.  [S3] Optional Tucker Tensor Decomposition for H2 coefficients to reduce parameters.
#     4.  [S3] Helper method to compute L1 loss for sparsity regularization.
#     """
#
#     def __init__(
#             self,
#             kq: float = 1.0,
#             learn_H0: bool = True,  # good
#             learn_H1: bool = True,  # transpose
#             learn_H2: bool = True,  # issue
#             ang_list_H0=None, ang_list_H1=None, ang_list_H2=[0, 2],
#             rad_list_H0=None, rad_list_H1=None, rad_list_H2=[2],
#             M: int = 8,
#             # [S3] Ranks for Tucker decomposition of H2 coeffs. Set to None to disable.
#             # Example: ranks={'i': 2, 'j': 2, 'a': 4, 'r': 4, 'm': 4}
#             # H2_ranks: dict = {'i': 2, 'j': 2, 'a': 4, 'r': 4, 'm': 4}
#             H2_ranks: dict = None
#     ):
#         super().__init__()
#         self.kq = kq
#         self.learn_H0, self.learn_H1, self.learn_H2 = learn_H0, learn_H1, learn_H2
#         self.global_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
#
#         # --- H0 parameters ---
#         if self.learn_H0:
#             self.ang_list_H0 = ang_list_H0 or [0];
#             self.rad_list_H0 = rad_list_H0 or [0];
#             self.M_H0 = M
#             nA0, nR0 = len(self.ang_list_H0), len(self.rad_list_H0)
#             self.alpha_env0 = nn.Parameter(torch.tensor(0.5))
#             self.alpha0 = nn.Parameter(torch.abs(torch.randn(nR0, self.M_H0)))
#             # [S1] Unified complex coefficients
#             shape0 = (nA0, nR0, self.M_H0)
#             self.c0_real = nn.Parameter(torch.zeros(shape0))
#             self.c0_imag = nn.Parameter(torch.zeros(shape0))
#
#         # --- H1 parameters ---
#         if self.learn_H1:
#             self.ang_list_H1 = ang_list_H1 or [1];
#             self.rad_list_H1 = rad_list_H1 or [1];
#             self.M_H1 = M
#             nA1, nR1 = len(self.ang_list_H1), len(self.rad_list_H1)
#             self.alpha_env1 = nn.Parameter(torch.tensor(0.5))
#             self.alpha1 = nn.Parameter(torch.abs(torch.randn(nR1, self.M_H1)))
#             # [S1] Unified complex coefficients
#             shape1 = (nA1, nR1, self.M_H1)
#             self.c1_real = nn.Parameter(torch.zeros(shape1))
#             self.c1_imag = nn.Parameter(torch.zeros(shape1))
#
#         # --- H2 parameters ---
#         if self.learn_H2:
#             self.ang_list_H2 = ang_list_H2 or [2];
#             self.rad_list_H2 = rad_list_H2 or [2];
#             self.M_H2 = M
#             self.H2_ranks = H2_ranks
#             nA2, nR2 = len(self.ang_list_H2), len(self.rad_list_H2)
#             self.alpha_env2 = nn.Parameter(torch.tensor(0.))
#             self.alpha2 = nn.Parameter(torch.abs(torch.randn(nR2, self.M_H2)))
#
#             if self.H2_ranks is None:
#                 # # No decomposition, standard full tensor
#                 shape2 = (nA2, nR2, self.M_H2)
#                 self.c2_real_11 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
#                 self.c2_imag_11 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
#
#                 self.c2_real_12 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
#                 self.c2_imag_12 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
#
#                 self.c2_real_22 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
#                 self.c2_imag_22 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
#
#                 # C_init = (1j / 4) * (self.kq ** 2)
#                 C_init = 1j
#
#                 # 找到角度索引
#                 idx_n0 = self.ang_list_H2.index(0)
#                 idx_n2 = self.ang_list_H2.index(2)
#                 # 找到径向索引
#                 m0 = 0
#                 idx_l2 = self.rad_list_H2.index(2)
#                 # with torch.no_grad():
#                 #     self.c2_real_11 *= 1j
#                 #     self.c2_imag_11 *= 1j
#                 #
#                 #     self.c2_real_12 *= 1j
#                 #     self.c2_imag_12 *= 1j
#                 #
#                 #     self.c2_real_22 *= 1j
#                 #     self.c2_imag_22 *= 1j
#                 with torch.no_grad():
#                     self.alpha2[idx_l2, :] = 0.0
#                     # 把模式 m0 的频率设成 kq
#                     self.alpha2[idx_l2, m0] = self.kq
#
#                     # H2_11 = H2_analytic * cos^2θ = H2 * (1 + cos2θ)/2
#                     # n=0 分量：+0.5*C_init
#                     self.c2_real_11[idx_n0, idx_l2, :] = 0.5 * C_init.real
#                     self.c2_imag_11[idx_n0, idx_l2, :] = 0.5 * C_init.imag
#                     # n=2 分量：+0.5*C_init
#                     self.c2_real_11[idx_n2, idx_l2, :] = 0.5 * C_init.real
#                     self.c2_imag_11[idx_n2, idx_l2, :] = 0.5 * C_init.imag
#
#                     # H2_22 = H2_analytic * sin^2θ = H2 * (1 - cos2θ)/2
#                     # n=0 分量：+0.5*C_init
#                     self.c2_real_22[idx_n0, idx_l2, :] = 0.5 * C_init.real
#                     self.c2_imag_22[idx_n0, idx_l2, :] = 0.5 * C_init.imag
#                     # n=2 分量：-0.5*C_init
#                     self.c2_real_22[idx_n2, idx_l2, :] = -0.5 * C_init.real
#                     self.c2_imag_22[idx_n2, idx_l2, :] = -0.5 * C_init.imag
#
#                     # H2_12 = H2_analytic * sinθ cosθ = H2 * (½ sin2θ)
#                     # sin2θ = (e^{i2θ} - e^{-i2θ})/(2i)，这里只保 n=2 imaginary 部分
#                     self.c2_real_12[idx_n2, idx_l2, :] = 0.0
#                     self.c2_imag_12[idx_n2, idx_l2, :] = 0.5 * C_init.real  # 对应 sin2θ/2
#
#                     # 对称
#                     self.c2_real_12[idx_n0, idx_l2, :] = 0.0
#                     self.c2_imag_12[idx_n0, idx_l2, :] = 0.0
#             else:
#                 # [S3] Tucker Decomposition Factors as parameters
#                 # Ranks for each dimension: (i, j, angular, radial, M)
#                 r = self.H2_ranks
#                 self.U_i = nn.Parameter(torch.randn(2, r['i']))
#                 self.U_j = nn.Parameter(torch.randn(2, r['j']))
#                 self.U_a = nn.Parameter(torch.randn(nA2, r['a']))
#                 self.U_r = nn.Parameter(torch.randn(nR2, r['r']))
#                 self.U_m = nn.Parameter(torch.randn(self.M_H2, r['m']))
#                 # Core tensor
#                 core_shape = (r['i'], r['j'], r['a'], r['r'], r['m'])
#                 self.G_core_real = nn.Parameter(torch.zeros(core_shape))
#                 self.G_core_imag = nn.Parameter(torch.zeros(core_shape))
#
#     def _bessel_J(self, l, x):
#         nu = 1.5
#
#         # nth-order Bessel J via recurrence
#         if l == 0:
#             return torch.special.bessel_j0(x)
#         elif l == 1:
#             return torch.special.bessel_j1(x)
#         else:
#             J0 = torch.special.bessel_j0(x)
#             J1 = torch.special.bessel_j1(x)
#             Jm2, Jm1 = J0, J1
#             for k in range(2, l + 1):
#                 Jl = (2 * (k - 1) / (x + 1e-8)) * Jm1 - Jm2
#                 Jm2, Jm1 = Jm1, Jl
#             return Jl
#
#     def _bessel_Y(self, l, x):
#         if l == 0:
#             return torch.special.bessel_y0(x)
#         elif l == 1:
#             return torch.special.bessel_y1(x)
#         else:
#             # 递推公式：Y_{n+1}(x) = (2n/x) * Y_n(x) - Y_{n-1}(x)
#             Y0 = torch.special.bessel_y0(x)
#             Y1 = torch.special.bessel_y1(x)
#             Ym2, Ym1 = Y0, Y1
#             for k in range(2, l + 1):
#                 Yl = (2 * (k - 1) / (x + 1e-8)) * Ym1 - Ym2
#                 Ym2, Ym1 = Ym1, Yl
#             return Yl
#
#     def forward(self, coords: torch.Tensor) -> torch.Tensor:
#         self.coords = coords
#         y, x = coords.unbind(-1)
#         r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
#         θ = torch.atan2(y, x)
#
#         B = coords.shape[0]
#         out = torch.zeros(B, 2, 2, dtype=torch.complex64, device=coords.device)
#
#         # --- H0 branch ---
#         if self.learn_H0:
#             H0 = self._learn_component(
#                 r, θ, self.ang_list_H0, self.rad_list_H0, self.M_H0,
#                 self.alpha_env0, self.alpha0, self.c0_real, self.c0_imag, is_scalar=True
#             )
#             out[:, 0, 0] += H0
#             out[:, 1, 1] += H0
#         else:  # Optional: add analytic form during inference
#             J0 = torch.special.bessel_j0(self.kq * r);
#             Y0 = self._bessel_Y(0, self.kq * r)
#             H0 = (1j / 4) * (self.kq ** 2) * (J0 + 1j * Y0)
#             out[:, 0, 0] += H0;
#             out[:, 1, 1] += H0
#
#         # --- H1 branch ---
#         if self.learn_H1:
#             H1 = self._learn_component(
#                 r, θ, self.ang_list_H1, self.rad_list_H1, self.M_H1,
#                 self.alpha_env1, self.alpha1, self.c1_real, self.c1_imag, is_scalar=True
#             )
#             out[:, 0, 0] += H1
#             out[:, 1, 1] += H1
#         else:
#             J1 = torch.special.bessel_j1(self.kq * r);
#             Y1 = self._bessel_Y(1, self.kq * r)
#             H1 = (1j / 4) * (-self.kq / r) * (J1 + 1j * Y1)
#             out[:, 0, 0] += H1;
#             out[:, 1, 1] += H1
#
#         # --- H2 branch ---
#         if self.learn_H2:
#
#
#
#             if self.H2_ranks is None:
#                 # Use standard full coefficients
#                 H2_11 = self._learn_component(
#                     r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
#                     self.alpha_env2, self.alpha2, self.c2_real_11, self.c2_imag_11, is_scalar=True, is_H2=True
#                 )
#                 H2_12 = self._learn_component(
#                     r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
#                     self.alpha_env2, self.alpha2, self.c2_real_12, self.c2_imag_12, is_scalar=True, is_H2=True
#                 )
#                 H2_22 = self._learn_component(
#                     r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
#                     self.alpha_env2, self.alpha2, self.c2_real_22, self.c2_imag_22, is_scalar=True, is_H2=True
#                 )
#                 row1 = torch.stack([H2_11, H2_12], dim=1)  # -> (N,2)
#                 row2 = torch.stack([H2_12, H2_22], dim=1)  # -> (N,2)
#                 H2 = torch.stack([row1, row2], dim=1)
#                 out += H2
#             else:
#                 G_core = self.G_core_real + 1j * self.G_core_imag
#
#                 # *** BUG FIX: Replaced incorrect einsum with a correct, robust einsum chain ***
#                 # This chain correctly reconstructs the full tensor from its Tucker factors.
#                 temp = torch.einsum('abcde,ia->ibcde', G_core, self.U_i.to(G_core.dtype))
#                 temp = torch.einsum('ibcde,jb->ijcde', temp, self.U_j.to(G_core.dtype))
#                 temp = torch.einsum('ijcde,kc->ijkde', temp, self.U_a.to(G_core.dtype))
#                 temp = torch.einsum('ijkde,ld->ijkle', temp, self.U_r.to(G_core.dtype))
#                 C2_recons = torch.einsum('ijkle,me->ijklm', temp, self.U_m.to(G_core.dtype))
#
#                 H2 = self._learn_component(
#                     r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
#                     self.alpha_env2, self.alpha2, C2_recons, None, is_scalar=False, coeffs_are_precomputed=True, is_H2=True
#                 )
#                 out += H2
#         else:
#             kr = self.kq * r;
#             J2 = 2 * torch.special.bessel_j1(kr) / (kr + 1e-8) - torch.special.bessel_j0(kr)
#             Y2 = self._bessel_Y(2, kr);
#             H2_analytic = (1j / 4) * (self.kq ** 2) * (J2 + 1j * Y2)
#             rhat_x = x / r;
#             rhat_y = y / r
#             out[:, 0, 0] += H2_analytic * (rhat_x ** 2);
#             out[:, 1, 1] += H2_analytic * (rhat_y ** 2)
#             out[:, 0, 1] += H2_analytic * (rhat_x * rhat_y);
#             out[:, 1, 0] += H2_analytic * (rhat_x * rhat_y);
#
#         return out * self.global_scale
#
#     # [S1] Unified learning function
#     def _learn_component(
#             self, r, theta, ang_list, rad_list, M,
#             alpha_env, alpha, c_real_or_precomputed, c_imag,
#             is_scalar: bool, coeffs_are_precomputed: bool = False, is_H2: bool = False
#     ):
#         B = r.shape[0]
#         env = r.pow(-alpha_env)
#
#         # --- 1. Get complex coefficients ---
#         if coeffs_are_precomputed:
#             C = c_real_or_precomputed
#         elif is_scalar:
#             C = c_real_or_precomputed + 1j * c_imag  # Shape (nA, nR, M)
#         else:  # Full matrix
#             C = c_real_or_precomputed + 1j * c_imag  # Shape (2, 2, nA, nR, M)
#
#         # --- 2. Compute basis functions ---
#         cosn = torch.cos(theta.unsqueeze(-1) * torch.tensor(ang_list, device=r.device))  # (B, nA)
#         sinn = torch.sin(theta.unsqueeze(-1) * torch.tensor(ang_list, device=r.device))  # (B, nA)
#         angular_basis = cosn + 1j* sinn  # Complex angular basis e^(i*n*theta), shape (B, nA)
#         angular_basis = angular_basis.to(dtype=torch.complex64)
#         arg = alpha.unsqueeze(0) * r.unsqueeze(1).unsqueeze(2)  # (B, nR, M)
#         radial_basis = torch.zeros(B, len(rad_list), M, dtype=torch.complex64, device=r.device)
#         for li, l in enumerate(rad_list):
#             if is_H2 and l == 2:
#                 kr = arg[:, li]
#                 J = 2 * torch.special.bessel_j1(kr) / (kr + 1e-8) - torch.special.bessel_j0(kr)
#                 Y = 2 * torch.special.bessel_y1(kr) / (kr + 1e-8) - torch.special.bessel_y0(kr)
#             else:
#                 J = self._bessel_J(l, arg[:, li])
#                 Y = self._bessel_Y(l, arg[:, li])
#             radial_basis[:, li, :] = J + 1j * Y
#
#         # --- 3. Contract coefficients with basis functions ---
#         # Using einsum for efficient and clear contraction
#         if is_scalar:
#             # B: batch, a: angular, r: radial, m: M-modes
#             # Coeffs C: (a, r, m), Ang_basis: (B, a), Rad_basis: (B, r, m)
#             # We want to sum over a, r, m for each batch item B
#             total = torch.einsum('arm, Ba, Brm -> B', C, angular_basis, radial_basis)
#         else:  # Full matrix
#             # i,j: matrix indices
#             total = torch.einsum('ijarm, Ba, Brm -> Bij', C, angular_basis, radial_basis)
#
#         return total * env.view(B, *([1] * (total.dim() - 1)))
#
#     # [S3] L1 regularization helper
#     def get_l1_loss(self):
#         """
#         Computes the L1 norm of all learnable coefficients.
#         Add this to your main loss in the training loop:
#         loss = main_loss + lambda_l1 * model.get_l1_loss()
#         """
#         l1_loss = 0.0
#         if self.learn_H0:
#             l1_loss += torch.abs(self.c0_real).sum() + torch.abs(self.c0_imag).sum()
#         if self.learn_H1:
#             l1_loss += torch.abs(self.c1_real).sum() + torch.abs(self.c1_imag).sum()
#         if self.learn_H2:
#             if self.H2_ranks is None:
#                 l1_loss += torch.abs(self.c2_real_11).sum() + torch.abs(self.c2_imag_11).sum()
#                 l1_loss += torch.abs(self.c2_real_12).sum() + torch.abs(self.c2_imag_12).sum()
#                 l1_loss += torch.abs(self.c2_real_22).sum() + torch.abs(self.c2_imag_22).sum()
#             else:
#                 # For decomposed tensors, regularize the factors and core
#                 l1_loss += torch.abs(self.G_core_real).sum() + torch.abs(self.G_core_imag).sum()
#                 for factor in [self.U_i, self.U_j, self.U_a, self.U_r, self.U_m]:
#                     l1_loss += torch.abs(factor).sum()
#         return l1_loss
#
#     def helmholtz_residual(self, coords: torch.Tensor) -> torch.Tensor:
#         """
#         Compute mean squared residual of (∇^2 + k^2) H_ij = 0
#         coords: (B,2) with requires_grad=True
#         """
#         coords = self.coords.clone().requires_grad_(True)
#         H = self.forward(coords)  # complex tensor (B,2,2)
#         k2 = self.kq ** 2
#         # accumulate squared residual
#         res = 0.0
#         for comp in (H.real, H.imag):
#             # comp: (B,2,2) real
#             B = comp.shape[0]
#             # loop over tensor components
#             for i in range(2):
#                 for j in range(2):
#                     Hij = comp[:, i, j]  # shape (B,)
#                     # first derivatives ∂H/∂x, ∂H/∂y
#                     grads = torch.autograd.grad(
#                         Hij.sum(), coords, create_graph=True
#                     )[0]  # shape (B,2)
#                     # build Laplacian
#                     lap = 0.0
#                     for d in range(2):
#                         g_d = grads[:, d]  # (B,)
#                         second = torch.autograd.grad(
#                             g_d.sum(), coords, create_graph=True
#                         )[0][:, d]  # (B,)
#                         lap = lap + second
#                     # residual = ∇²Hij + k² Hij
#                     res = res + ((lap + k2 * Hij) ** 2).sum()
#         # return mean over all coords & components
#         return res / coords.numel()
#
#     def double_curl_residual(self, coords: torch.Tensor) -> torch.Tensor:
#         """
#         计算 double curl PDE 残差损失: (∇×∇×G + k^2 G) = 0
#         coords: (B,2) with requires_grad=True
#         """
#         coords = self.coords.clone().requires_grad_(True)
#         G = self.forward(coords)  # (B, 2, 2)
#         k2 = self.kq ** 2
#         res = 0.0
#         for comp in (G.real, G.imag):
#             # 对每一列（即每个输出分量）分别做double curl
#             for j in range(2):
#                 Gj = comp[:, :, j]  # (B, 2) 第j列
#                 # 1. divergence: div(Gj) = dGx/dx + dGy/dy
#                 div = 0.0
#                 for i in range(2):
#                     grad = torch.autograd.grad(Gj[:, i].sum(), coords, create_graph=True)[0][:, i]
#                     div = div + grad  # (B,)
#                 # 2. grad(div(Gj)): shape (B,2)
#                 grad_div = torch.stack([
#                     torch.autograd.grad(div.sum(), coords, create_graph=True)[0][:, 0],  # d(div)/dx
#                     torch.autograd.grad(div.sum(), coords, create_graph=True)[0][:, 1],  # d(div)/dy
#                 ], dim=1)  # (B,2)
#                 # 3. Laplacian of Gj: ΔGj_i = d²Gj_i/dx² + d²Gj_i/dy²
#                 lap = torch.zeros_like(Gj)
#                 for i in range(2):
#                     grad_i = torch.autograd.grad(Gj[:, i].sum(), coords, create_graph=True)[0]
#                     for d in range(2):
#                         second = torch.autograd.grad(grad_i[:, d].sum(), coords, create_graph=True)[0][:, d]
#                         lap[:, i] = lap[:, i] + second
#                 # 4. double curl: grad_div - lap
#                 double_curl = grad_div - lap  # (B,2)
#                 # 5. PDE残差: double_curl + k^2 * Gj
#                 pde_res = double_curl + k2 * Gj  # (B,2)
#                 res = res + (pde_res ** 2).sum()
#         return res / coords.numel()
#
#     def l2norm_loss(self,coords: torch.Tensor) -> torch.Tensor:
#         coords = self.coords.clone().requires_grad_(True)
#         G = self.forward(coords)
#         return torch.sum(torch.sum(torch.real(G.conj() * G)))
#
#     def divergence_loss(self, coords: torch.Tensor) -> torch.Tensor:
#         """
#         计算无散度损失：div(G) = ∂G_{ij}/∂x_i，返回L2范数均值。
#         coords: (B,2) with requires_grad=True
#         """
#         coords = coords.clone().requires_grad_(True)
#         G = self.forward(coords)  # (B, 2, 2)
#         B = G.shape[0]
#         div = torch.zeros(B, 2, dtype=G.dtype, device=coords.device)  # (B, 2)
#         for j in range(2):  # 对每一列
#             # G_0j, G_1j 分别对x和y求偏导
#             grads = torch.autograd.grad(G[:, 0, j].sum().real, coords, create_graph=True)[0][:, 0] \
#                   + torch.autograd.grad(G[:, 1, j].sum().real, coords, create_graph=True)[0][:, 1] \
#                   + torch.autograd.grad(G[:, 0, j].sum().imag, coords, create_graph=True)[0][:, 0] \
#                   + torch.autograd.grad(G[:, 1, j].sum().imag, coords, create_graph=True)[0][:, 1]
#             div[:, j] = grads
#         # L2范数
#         loss = (div.abs() ** 2).sum() / div.numel()
#         return loss
#
#
#
import torch
import torch.nn as nn
import json
from math import pi

class BesselFourier2x2_v2(nn.Module):
    """
    IO-compatible:
      - same class/forward signature and output (B,2,2) complex.
      - __init__ accepts your old args but ignores them internally (uses its own tiny dictionary).

    New:
      - Physics-aware polar atoms (H0,H1,H2) + tiny BOX corrector.
      - Two regularization paths:
          * group-lasso (families + per-order groups)
          * individual-lasso (each atom)
      - Activity probes (family-level and per-atom) for "hottest" tracking.
    """

    def __init__(self,
        kq: float = 1.0,
        learn_H0: bool = True, learn_H1: bool = True, learn_H2: bool = True,
        ang_list_H0=None, ang_list_H1=None, ang_list_H2=[0,2],
        rad_list_H0=None, rad_list_H1=None, rad_list_H2=[2],
        M: int = 8,
        H2_ranks: dict = None
    ):
        super().__init__()
        # --- internal defaults (we ignore ctor config to keep things tiny & robust) ---
        self.kq = float(kq) if kq is not None else 8.0
        self.learn_H0 = True
        self.learn_H1 = True   # off by default; enable if you need dipole
        self.learn_H2 = True

        # tiny angular/radial sets
        self.ang_list_H0 = [0]
        self.ang_list_H1 = [1]
        self.ang_list_H2 = [0, 2]
        self.rad_list_H0 = [0]
        self.rad_list_H1 = [1]
        self.rad_list_H2 = [2]
        self.M_H0 = self.M_H1 = self.M_H2 = 4

        self.global_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # envelopes
        self.alpha_env0 = nn.Parameter(torch.tensor(0.25))
        self.alpha_env1 = nn.Parameter(torch.tensor(0.25))
        self.alpha_env2 = nn.Parameter(torch.tensor(0.00))

        def init_freq(nR, M, k):
            a = torch.full((nR, M), max(k, 1e-3))
            if M > 1: a[:, 1:] = k * (1.0 + 0.1 * torch.randn(nR, M - 1))
            return nn.Parameter(a.abs())

        # H0
        if self.learn_H0:
            self.alpha0 = init_freq(len(self.rad_list_H0), self.M_H0, self.kq)
            self.c0_real = nn.Parameter(torch.zeros(len(self.ang_list_H0), len(self.rad_list_H0), self.M_H0))
            self.c0_imag = nn.Parameter(torch.zeros_like(self.c0_real))
        # H1
        if self.learn_H1:
            self.alpha1 = init_freq(len(self.rad_list_H1), self.M_H1, self.kq)
            self.c1_real = nn.Parameter(torch.zeros(len(self.ang_list_H1), len(self.rad_list_H1), self.M_H1))
            self.c1_imag = nn.Parameter(torch.zeros_like(self.c1_real))
        # H2 (structured tensor from a scalar Hankel amplitude)
        if self.learn_H2:
            self.alpha2 = init_freq(len(self.rad_list_H2), self.M_H2, self.kq)
            self.c2_real = nn.Parameter(torch.zeros(len(self.ang_list_H2), len(self.rad_list_H2), self.M_H2))
            self.c2_imag = nn.Parameter(torch.zeros_like(self.c2_real))

        # BOX corrector (very few modes)
        self.use_box = True
        self.Lx = 1.0; self.Ly = 1.0
        self.box_modes = [(1,1), (2,1), (1,2)]
        self.box_coeff_re = nn.Parameter(torch.zeros(len(self.box_modes), 2, 2))
        self.box_coeff_im = nn.Parameter(torch.zeros_like(self.box_coeff_re))

        # <<< NEW: small normal init for coeffs >>>
        self.reset_parameters(std=1e-3)

        # --- NEW ---

    def reset_parameters(self, std: float = 1e-3):
        """
        Initialize complex coefficients with small normal noise so gradients
        are non-zero at step 0, avoiding the all-zero stationary point.
        """
        with torch.no_grad():
            if self.learn_H0:
                self.c0_real.normal_(mean=0.0, std=std)
                self.c0_imag.normal_(mean=0.0, std=std)
            if self.learn_H1:
                self.c1_real.normal_(mean=0.0, std=std)
                self.c1_imag.normal_(mean=0.0, std=std)
            if self.learn_H2:
                self.c2_real.normal_(mean=0.0, std=std)
                self.c2_imag.normal_(mean=0.0, std=std)

            if self.use_box:
                self.box_coeff_re.normal_(mean=0.0, std=std)
                self.box_coeff_im.normal_(mean=0.0, std=std)

            # keep global scale ~1
            self.global_scale.clamp_(min=1e-3, max=10.0)

    # ---- Hankel helpers ----
    @staticmethod
    def _J(n, z):
        if n == 0: return torch.special.bessel_j0(z)
        if n == 1: return torch.special.bessel_j1(z)
        z = z + 1e-8
        Jm2, Jm1 = torch.special.bessel_j0(z), torch.special.bessel_j1(z)
        for k in range(2, n+1):
            Jk = (2*(k-1)/z)*Jm1 - Jm2
            Jm2, Jm1 = Jm1, Jk
        return Jm1
    @staticmethod
    def _Y(n, z):
        if n == 0: return torch.special.bessel_y0(z)
        if n == 1: return torch.special.bessel_y1(z)
        z = z + 1e-8
        Ym2, Ym1 = torch.special.bessel_y0(z), torch.special.bessel_y1(z)
        for k in range(2, n+1):
            Yk = (2*(k-1)/z)*Ym1 - Ym2
            Ym2, Ym1 = Ym1, Yk
        return Ym1
    @staticmethod
    def _H1(n, z):
        return BesselFourier2x2._J(n, z) + 1j * BesselFourier2x2._Y(n, z)

    # ---- polar block ----
    def _polar_sum(self, r, theta, ang_list, rad_list, M, alpha_env, alpha, c_re, c_im, hankel_order):
        B = r.shape[0]
        env = (r + 1e-6).pow(-torch.clamp(alpha_env, -1.0, 2.0))
        n_vec = torch.tensor(ang_list, device=r.device, dtype=r.dtype)
        ang = torch.cos(theta.unsqueeze(-1) * n_vec) + 1j * torch.sin(theta.unsqueeze(-1) * n_vec)  # (B,nA)

        nR = len(rad_list)
        R = torch.zeros(B, nR, M, dtype=torch.complex64, device=r.device)
        for li, _l in enumerate(rad_list):
            z = torch.clamp(alpha[li].unsqueeze(0) * r.unsqueeze(-1), min=1e-6)
            R[:, li, :] = self._H1(hankel_order, z)

        C = (c_re + 1j * c_im).to(R.dtype)
        total = torch.einsum('arm,Ba,Brm->B', C, ang.to(R.dtype), R)
        return total * env

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        y, x = coords.unbind(-1)
        r = torch.sqrt(x*x + y*y + 1e-8)
        th = torch.atan2(y, x)
        B = coords.shape[0]
        out = torch.zeros(B, 2, 2, dtype=torch.complex64, device=coords.device)

        if self.learn_H0:
            H0 = self._polar_sum(r, th, self.ang_list_H0, self.rad_list_H0, self.M_H0,
                                 self.alpha_env0, self.alpha0, self.c0_real, self.c0_imag, hankel_order=0)
            out[:,0,0] += H0; out[:,1,1] += H0

        if self.learn_H1:
            H1 = self._polar_sum(r, th, self.ang_list_H1, self.rad_list_H1, self.M_H1,
                                 self.alpha_env1, self.alpha1, self.c1_real, self.c1_imag, hankel_order=1)
            out[:,0,0] += H1; out[:,1,1] += H1

        if self.learn_H2:
            A = self._polar_sum(r, th, self.ang_list_H2, self.rad_list_H2, self.M_H2,
                                self.alpha_env2, self.alpha2, self.c2_real, self.c2_imag, hankel_order=2)
            rinv = 1.0/(r+1e-6); cx, cy = x*rinv, y*rinv
            cos2t = cx*cx - cy*cy
            sin2t = 2.0*cx*cy
            out[:,0,0] += 0.5*A*(1.0 + cos2t)
            out[:,1,1] += 0.5*A*(1.0 - cos2t)
            off = 0.5*A*sin2t
            out[:,0,1] += off; out[:,1,0] += off

        if self.use_box:
            # relative coords assumed in [-0.5,0.5]^2
            X = (x + 0.5*self.Lx)/self.Lx
            Y = (y + 0.5*self.Ly)/self.Ly
            for idx,(m,n) in enumerate(self.box_modes):
                mode = torch.sin(m*pi*X)*torch.sin(n*pi*Y)  # (B,)
                C = (self.box_coeff_re[idx] + 1j*self.box_coeff_im[idx]).to(out.dtype)
                out = out + mode.view(-1,1,1) * C

        return out * self.global_scale

    # --------- regularizers ----------
    def lasso_individual(self):
        """L1 over every atom coefficient (individual sparsity)."""
        l1 = 0.0
        if self.learn_H0: l1 += self.c0_real.abs().sum() + self.c0_imag.abs().sum()
        if self.learn_H1: l1 += self.c1_real.abs().sum() + self.c1_imag.abs().sum()
        if self.learn_H2: l1 += self.c2_real.abs().sum() + self.c2_imag.abs().sum()
        if self.use_box:
            l1 += self.box_coeff_re.abs().sum() + self.box_coeff_im.abs().sum()
        return l1

    def lasso_group_family(self):
        """Group-lasso over families + per-order groups."""
        norms = []
        if self.learn_H0:
            norms += [torch.linalg.vector_norm(torch.cat([self.c0_real.flatten(), self.c0_imag.flatten()]))]
            # per-order
            for ai,_ in enumerate(self.ang_list_H0):
                norms += [torch.linalg.vector_norm(torch.cat([self.c0_real[ai].flatten(), self.c0_imag[ai].flatten()]))]
        if self.learn_H1:
            norms += [torch.linalg.vector_norm(torch.cat([self.c1_real.flatten(), self.c1_imag.flatten()]))]
            for ai,_ in enumerate(self.ang_list_H1):
                norms += [torch.linalg.vector_norm(torch.cat([self.c1_real[ai].flatten(), self.c1_imag[ai].flatten()]))]
        if self.learn_H2:
            norms += [torch.linalg.vector_norm(torch.cat([self.c2_real.flatten(), self.c2_imag.flatten()]))]
            for ai,_ in enumerate(self.ang_list_H2):
                norms += [torch.linalg.vector_norm(torch.cat([self.c2_real[ai].flatten(), self.c2_imag[ai].flatten()]))]
        if self.use_box:
            norms += [torch.linalg.vector_norm(torch.cat([self.box_coeff_re.flatten(), self.box_coeff_im.flatten()]))]
        return sum(norms)

    # --------- activity probes (for "hottest" JSON logging) ----------
    def activity_family(self):
        """Returns dict: {'H0': float, 'H1': ..., 'H2': ..., 'BOX': float}."""
        d = {}
        if self.learn_H0:
            d['H0'] = float(torch.sqrt(self.c0_real.pow(2).sum() + self.c0_imag.pow(2).sum()).detach().cpu())
        if self.learn_H1:
            d['H1'] = float(torch.sqrt(self.c1_real.pow(2).sum() + self.c1_imag.pow(2).sum()).detach().cpu())
        if self.learn_H2:
            d['H2'] = float(torch.sqrt(self.c2_real.pow(2).sum() + self.c2_imag.pow(2).sum()).detach().cpu())
        if self.use_box:
            d['BOX'] = float(torch.sqrt(self.box_coeff_re.pow(2).sum() + self.box_coeff_im.pow(2).sum()).detach().cpu())
        return d

    def activity_individual(self):
        """
        Returns fine-grained dict: one entry per atom, e.g.:
          'H0[n=0,l=0,m=0]' : norm, ...
          'H2[n=2,l=2,m=1]' : norm, ...
          'BOX[(m,n)=(1,2)]' : Frobenius norm of its 2x2 coeff.
        """
        d = {}
        if self.learn_H0:
            for ai,n in enumerate(self.ang_list_H0):
                for li,l in enumerate(self.rad_list_H0):
                    for m in range(self.M_H0):
                        v = torch.sqrt(self.c0_real[ai,li,m]**2 + self.c0_imag[ai,li,m]**2)
                        d[f'H0[n={n},l={l},m={m}]'] = float(v.detach().cpu())
        if self.learn_H1:
            for ai,n in enumerate(self.ang_list_H1):
                for li,l in enumerate(self.rad_list_H1):
                    for m in range(self.M_H1):
                        v = torch.sqrt(self.c1_real[ai,li,m]**2 + self.c1_imag[ai,li,m]**2)
                        d[f'H1[n={n},l={l},m={m}]'] = float(v.detach().cpu())
        if self.learn_H2:
            for ai,n in enumerate(self.ang_list_H2):
                for li,l in enumerate(self.rad_list_H2):
                    for m in range(self.M_H2):
                        v = torch.sqrt(self.c2_real[ai,li,m]**2 + self.c2_imag[ai,li,m]**2)
                        d[f'H2[n={n},l={l},m={m}]'] = float(v.detach().cpu())
        if self.use_box:
            for idx,(m1,n1) in enumerate(self.box_modes):
                C = self.box_coeff_re[idx] + 1j*self.box_coeff_im[idx]
                d[f'BOX[(m={m1},n={n1})]'] = float(torch.linalg.vector_norm(C).detach().cpu())
        return d

    # --------- physics losses (unchanged APIs) ----------
    def helmholtz_residual(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        G = self.forward(coords)
        k2 = self.kq**2
        res = 0.0
        for comp in (G.real, G.imag):
            for i in range(2):
                for j in range(2):
                    Hij = comp[:, i, j]
                    grad = torch.autograd.grad(Hij.sum(), coords, create_graph=True)[0]
                    lap = 0.0
                    for d in range(2):
                        second = torch.autograd.grad(grad[:, d].sum(), coords, create_graph=True)[0][:, d]
                        lap = lap + second
                    res = res + ((lap + k2*Hij)**2).mean()
        return res

    def double_curl_residual(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        G = self.forward(coords)
        k2 = self.kq**2
        res = 0.0
        for comp in (G.real, G.imag):
            for j in range(2):
                Gj = comp[:, :, j]
                dGx_dx = torch.autograd.grad(Gj[:,0].sum(), coords, create_graph=True)[0][:,0]
                dGy_dy = torch.autograd.grad(Gj[:,1].sum(), coords, create_graph=True)[0][:,1]
                div = dGx_dx + dGy_dy
                gdiv = torch.autograd.grad(div.sum(), coords, create_graph=True)[0]
                lap = torch.zeros_like(Gj)
                for i in range(2):
                    gi = torch.autograd.grad(Gj[:,i].sum(), coords, create_graph=True)[0]
                    lap[:,i] = (torch.autograd.grad(gi[:,0].sum(), coords, create_graph=True)[0][:,0] +
                                torch.autograd.grad(gi[:,1].sum(), coords, create_graph=True)[0][:,1])
                res = res + ((gdiv - lap + k2*Gj)**2).mean()
        return res

    def l2norm_loss(self, coords: torch.Tensor) -> torch.Tensor:
        G = self.forward(coords)
        return torch.real((G.conj()*G).sum()) / coords.shape[0]

    def divergence_loss(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        G = self.forward(coords)
        div_norm = 0.0
        for j in range(2):
            for part in (G.real, G.imag):
                d0dx = torch.autograd.grad(part[:,0,j].sum(), coords, create_graph=True)[0][:,0]
                d1dy = torch.autograd.grad(part[:,1,j].sum(), coords, create_graph=True)[0][:,1]
                div = d0dx + d1dy
                div_norm += (div**2).mean()
        return div_norm




class BesselFourier2x2_v1(nn.Module):
    """
    V2: 2x2 Green's tensor with preserved physical interpretability (H0, H1, H2).
    Implements optimizations from Strategy 1 and 3:
    1.  [S1] Unified complex coefficient structure for all learnable branches.
    2.  [S1] Unified internal learning function `_learn_component` to reduce code redundancy.
    3.  [S3] Optional Tucker Tensor Decomposition for H2 coefficients to reduce parameters.
    4.  [S3] Helper method to compute L1 loss for sparsity regularization.
    """

    def __init__(
            self,
            kq: float = 1.0,
            learn_H0: bool = True,  # good
            learn_H1: bool = True,  # transpose
            learn_H2: bool = True,  # issue
            ang_list_H0=None, ang_list_H1=None, ang_list_H2=[0, 2],
            rad_list_H0=None, rad_list_H1=None, rad_list_H2=[2],
            M: int = 8,
            # [S3] Ranks for Tucker decomposition of H2 coeffs. Set to None to disable.
            # Example: ranks={'i': 2, 'j': 2, 'a': 4, 'r': 4, 'm': 4}
            # H2_ranks: dict = {'i': 2, 'j': 2, 'a': 4, 'r': 4, 'm': 4}
            H2_ranks: dict = None
    ):
        super().__init__()
        self.kq = kq
        self.learn_H0, self.learn_H1, self.learn_H2 = learn_H0, learn_H1, learn_H2
        self.global_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # --- H0 parameters ---
        if self.learn_H0:
            self.ang_list_H0 = ang_list_H0 or [0];
            self.rad_list_H0 = rad_list_H0 or [0];
            self.M_H0 = M
            nA0, nR0 = len(self.ang_list_H0), len(self.rad_list_H0)
            self.alpha_env0 = nn.Parameter(torch.tensor(0.5))
            self.alpha0 = nn.Parameter(torch.abs(torch.randn(nR0, self.M_H0)))
            # [S1] Unified complex coefficients
            shape0 = (nA0, nR0, self.M_H0)
            self.c0_real = nn.Parameter(torch.zeros(shape0))
            self.c0_imag = nn.Parameter(torch.zeros(shape0))

        # --- H1 parameters ---
        if self.learn_H1:
            self.ang_list_H1 = ang_list_H1 or [1];
            self.rad_list_H1 = rad_list_H1 or [1];
            self.M_H1 = M
            nA1, nR1 = len(self.ang_list_H1), len(self.rad_list_H1)
            self.alpha_env1 = nn.Parameter(torch.tensor(0.5))
            self.alpha1 = nn.Parameter(torch.abs(torch.randn(nR1, self.M_H1)))
            # [S1] Unified complex coefficients
            shape1 = (nA1, nR1, self.M_H1)
            self.c1_real = nn.Parameter(torch.zeros(shape1))
            self.c1_imag = nn.Parameter(torch.zeros(shape1))

        # --- H2 parameters ---
        if self.learn_H2:
            self.ang_list_H2 = ang_list_H2 or [2];
            self.rad_list_H2 = rad_list_H2 or [2];
            self.M_H2 = M
            self.H2_ranks = H2_ranks
            nA2, nR2 = len(self.ang_list_H2), len(self.rad_list_H2)
            self.alpha_env2 = nn.Parameter(torch.tensor(0.))
            self.alpha2 = nn.Parameter(torch.abs(torch.randn(nR2, self.M_H2)))

            if self.H2_ranks is None:
                # # No decomposition, standard full tensor
                shape2 = (nA2, nR2, self.M_H2)
                self.c2_real_11 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
                self.c2_imag_11 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))

                self.c2_real_12 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
                self.c2_imag_12 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))

                self.c2_real_22 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))
                self.c2_imag_22 = nn.Parameter(torch.ones(shape2,dtype=torch.cfloat))

                # C_init = (1j / 4) * (self.kq ** 2)
                C_init = 1j

                # 找到角度索引
                idx_n0 = self.ang_list_H2.index(0)
                idx_n2 = self.ang_list_H2.index(2)
                # 找到径向索引
                m0 = 0
                idx_l2 = self.rad_list_H2.index(2)
                # with torch.no_grad():
                #     self.c2_real_11 *= 1j
                #     self.c2_imag_11 *= 1j
                #
                #     self.c2_real_12 *= 1j
                #     self.c2_imag_12 *= 1j
                #
                #     self.c2_real_22 *= 1j
                #     self.c2_imag_22 *= 1j
                with torch.no_grad():
                    self.alpha2[idx_l2, :] = 0.0
                    # 把模式 m0 的频率设成 kq
                    self.alpha2[idx_l2, m0] = self.kq

                    # H2_11 = H2_analytic * cos^2θ = H2 * (1 + cos2θ)/2
                    # n=0 分量：+0.5*C_init
                    self.c2_real_11[idx_n0, idx_l2, :] = 0.5 * C_init.real
                    self.c2_imag_11[idx_n0, idx_l2, :] = 0.5 * C_init.imag
                    # n=2 分量：+0.5*C_init
                    self.c2_real_11[idx_n2, idx_l2, :] = 0.5 * C_init.real
                    self.c2_imag_11[idx_n2, idx_l2, :] = 0.5 * C_init.imag

                    # H2_22 = H2_analytic * sin^2θ = H2 * (1 - cos2θ)/2
                    # n=0 分量：+0.5*C_init
                    self.c2_real_22[idx_n0, idx_l2, :] = 0.5 * C_init.real
                    self.c2_imag_22[idx_n0, idx_l2, :] = 0.5 * C_init.imag
                    # n=2 分量：-0.5*C_init
                    self.c2_real_22[idx_n2, idx_l2, :] = -0.5 * C_init.real
                    self.c2_imag_22[idx_n2, idx_l2, :] = -0.5 * C_init.imag

                    # H2_12 = H2_analytic * sinθ cosθ = H2 * (½ sin2θ)
                    # sin2θ = (e^{i2θ} - e^{-i2θ})/(2i)，这里只保 n=2 imaginary 部分
                    self.c2_real_12[idx_n2, idx_l2, :] = 0.0
                    self.c2_imag_12[idx_n2, idx_l2, :] = 0.5 * C_init.real  # 对应 sin2θ/2

                    # 对称
                    self.c2_real_12[idx_n0, idx_l2, :] = 0.0
                    self.c2_imag_12[idx_n0, idx_l2, :] = 0.0
            else:
                # [S3] Tucker Decomposition Factors as parameters
                # Ranks for each dimension: (i, j, angular, radial, M)
                r = self.H2_ranks
                self.U_i = nn.Parameter(torch.randn(2, r['i']))
                self.U_j = nn.Parameter(torch.randn(2, r['j']))
                self.U_a = nn.Parameter(torch.randn(nA2, r['a']))
                self.U_r = nn.Parameter(torch.randn(nR2, r['r']))
                self.U_m = nn.Parameter(torch.randn(self.M_H2, r['m']))
                # Core tensor
                core_shape = (r['i'], r['j'], r['a'], r['r'], r['m'])
                self.G_core_real = nn.Parameter(torch.zeros(core_shape))
                self.G_core_imag = nn.Parameter(torch.zeros(core_shape))

    def _bessel_J(self, l, x):
        nu = 1.5

        # nth-order Bessel J via recurrence
        if l == 0:
            return torch.special.bessel_j0(x)
        elif l == 1:
            return torch.special.bessel_j1(x)
        else:
            J0 = torch.special.bessel_j0(x)
            J1 = torch.special.bessel_j1(x)
            Jm2, Jm1 = J0, J1
            for k in range(2, l + 1):
                Jl = (2 * (k - 1) / (x + 1e-8)) * Jm1 - Jm2
                Jm2, Jm1 = Jm1, Jl
            return Jl

    def _bessel_Y(self, l, x):
        if l == 0:
            return torch.special.bessel_y0(x)
        elif l == 1:
            return torch.special.bessel_y1(x)
        else:
            # 递推公式：Y_{n+1}(x) = (2n/x) * Y_n(x) - Y_{n-1}(x)
            Y0 = torch.special.bessel_y0(x)
            Y1 = torch.special.bessel_y1(x)
            Ym2, Ym1 = Y0, Y1
            for k in range(2, l + 1):
                Yl = (2 * (k - 1) / (x + 1e-8)) * Ym1 - Ym2
                Ym2, Ym1 = Ym1, Yl
            return Yl

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        self.coords = coords
        y, x = coords.unbind(-1)
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        θ = torch.atan2(y, x)

        B = coords.shape[0]
        out = torch.zeros(B, 2, 2, dtype=torch.complex64, device=coords.device)

        # --- H0 branch ---
        if self.learn_H0:
            H0 = self._learn_component(
                r, θ, self.ang_list_H0, self.rad_list_H0, self.M_H0,
                self.alpha_env0, self.alpha0, self.c0_real, self.c0_imag, is_scalar=True
            )
            out[:, 0, 0] += H0
            out[:, 1, 1] += H0
        else:  # Optional: add analytic form during inference
            J0 = torch.special.bessel_j0(self.kq * r);
            Y0 = self._bessel_Y(0, self.kq * r)
            H0 = (1j / 4) * (self.kq ** 2) * (J0 + 1j * Y0)
            out[:, 0, 0] += H0;
            out[:, 1, 1] += H0

        # --- H1 branch ---
        if self.learn_H1:
            H1 = self._learn_component(
                r, θ, self.ang_list_H1, self.rad_list_H1, self.M_H1,
                self.alpha_env1, self.alpha1, self.c1_real, self.c1_imag, is_scalar=True
            )
            out[:, 0, 0] += H1
            out[:, 1, 1] += H1
        else:
            J1 = torch.special.bessel_j1(self.kq * r);
            Y1 = self._bessel_Y(1, self.kq * r)
            H1 = (1j / 4) * (-self.kq / r) * (J1 + 1j * Y1)
            out[:, 0, 0] += H1;
            out[:, 1, 1] += H1

        # --- H2 branch ---
        if self.learn_H2:



            if self.H2_ranks is None:
                # Use standard full coefficients
                H2_11 = self._learn_component(
                    r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
                    self.alpha_env2, self.alpha2, self.c2_real_11, self.c2_imag_11, is_scalar=True, is_H2=True
                )
                H2_12 = self._learn_component(
                    r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
                    self.alpha_env2, self.alpha2, self.c2_real_12, self.c2_imag_12, is_scalar=True, is_H2=True
                )
                H2_22 = self._learn_component(
                    r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
                    self.alpha_env2, self.alpha2, self.c2_real_22, self.c2_imag_22, is_scalar=True, is_H2=True
                )
                row1 = torch.stack([H2_11, H2_12], dim=1)  # -> (N,2)
                row2 = torch.stack([H2_12, H2_22], dim=1)  # -> (N,2)
                H2 = torch.stack([row1, row2], dim=1)
                out += H2
            else:
                G_core = self.G_core_real + 1j * self.G_core_imag

                # *** BUG FIX: Replaced incorrect einsum with a correct, robust einsum chain ***
                # This chain correctly reconstructs the full tensor from its Tucker factors.
                temp = torch.einsum('abcde,ia->ibcde', G_core, self.U_i.to(G_core.dtype))
                temp = torch.einsum('ibcde,jb->ijcde', temp, self.U_j.to(G_core.dtype))
                temp = torch.einsum('ijcde,kc->ijkde', temp, self.U_a.to(G_core.dtype))
                temp = torch.einsum('ijkde,ld->ijkle', temp, self.U_r.to(G_core.dtype))
                C2_recons = torch.einsum('ijkle,me->ijklm', temp, self.U_m.to(G_core.dtype))

                H2 = self._learn_component(
                    r, θ, self.ang_list_H2, self.rad_list_H2, self.M_H2,
                    self.alpha_env2, self.alpha2, C2_recons, None, is_scalar=False, coeffs_are_precomputed=True, is_H2=True
                )
                out += H2
        else:
            kr = self.kq * r;
            J2 = 2 * torch.special.bessel_j1(kr) / (kr + 1e-8) - torch.special.bessel_j0(kr)
            Y2 = self._bessel_Y(2, kr);
            H2_analytic = (1j / 4) * (self.kq ** 2) * (J2 + 1j * Y2)
            rhat_x = x / r;
            rhat_y = y / r
            out[:, 0, 0] += H2_analytic * (rhat_x ** 2);
            out[:, 1, 1] += H2_analytic * (rhat_y ** 2)
            out[:, 0, 1] += H2_analytic * (rhat_x * rhat_y);
            out[:, 1, 0] += H2_analytic * (rhat_x * rhat_y);

        return out * self.global_scale

    # [S1] Unified learning function
    def _learn_component(
            self, r, theta, ang_list, rad_list, M,
            alpha_env, alpha, c_real_or_precomputed, c_imag,
            is_scalar: bool, coeffs_are_precomputed: bool = False, is_H2: bool = False
    ):
        B = r.shape[0]
        env = r.pow(-alpha_env)

        # --- 1. Get complex coefficients ---
        if coeffs_are_precomputed:
            C = c_real_or_precomputed
        elif is_scalar:
            C = c_real_or_precomputed + 1j * c_imag  # Shape (nA, nR, M)
        else:  # Full matrix
            C = c_real_or_precomputed + 1j * c_imag  # Shape (2, 2, nA, nR, M)

        # --- 2. Compute basis functions ---
        cosn = torch.cos(theta.unsqueeze(-1) * torch.tensor(ang_list, device=r.device))  # (B, nA)
        sinn = torch.sin(theta.unsqueeze(-1) * torch.tensor(ang_list, device=r.device))  # (B, nA)
        angular_basis = cosn + 1j* sinn  # Complex angular basis e^(i*n*theta), shape (B, nA)
        angular_basis = angular_basis.to(dtype=torch.complex64)
        arg = alpha.unsqueeze(0) * r.unsqueeze(1).unsqueeze(2)  # (B, nR, M)
        radial_basis = torch.zeros(B, len(rad_list), M, dtype=torch.complex64, device=r.device)
        for li, l in enumerate(rad_list):
            if is_H2 and l == 2:
                kr = arg[:, li]
                J = 2 * torch.special.bessel_j1(kr) / (kr + 1e-8) - torch.special.bessel_j0(kr)
                Y = 2 * torch.special.bessel_y1(kr) / (kr + 1e-8) - torch.special.bessel_y0(kr)
            else:
                J = self._bessel_J(l, arg[:, li])
                Y = self._bessel_Y(l, arg[:, li])
            radial_basis[:, li, :] = J + 1j * Y

        # --- 3. Contract coefficients with basis functions ---
        # Using einsum for efficient and clear contraction
        if is_scalar:
            # B: batch, a: angular, r: radial, m: M-modes
            # Coeffs C: (a, r, m), Ang_basis: (B, a), Rad_basis: (B, r, m)
            # We want to sum over a, r, m for each batch item B
            total = torch.einsum('arm, Ba, Brm -> B', C, angular_basis, radial_basis)
        else:  # Full matrix
            # i,j: matrix indices
            total = torch.einsum('ijarm, Ba, Brm -> Bij', C, angular_basis, radial_basis)

        return total * env.view(B, *([1] * (total.dim() - 1)))

    # [S3] L1 regularization helper
    def get_l1_loss(self):
        """
        Computes the L1 norm of all learnable coefficients.
        Add this to your main loss in the training loop:
        loss = main_loss + lambda_l1 * model.get_l1_loss()
        """
        l1_loss = 0.0
        if self.learn_H0:
            l1_loss += torch.abs(self.c0_real).sum() + torch.abs(self.c0_imag).sum()
        if self.learn_H1:
            l1_loss += torch.abs(self.c1_real).sum() + torch.abs(self.c1_imag).sum()
        if self.learn_H2:
            if self.H2_ranks is None:
                l1_loss += torch.abs(self.c2_real_11).sum() + torch.abs(self.c2_imag_11).sum()
                l1_loss += torch.abs(self.c2_real_12).sum() + torch.abs(self.c2_imag_12).sum()
                l1_loss += torch.abs(self.c2_real_22).sum() + torch.abs(self.c2_imag_22).sum()
            else:
                # For decomposed tensors, regularize the factors and core
                l1_loss += torch.abs(self.G_core_real).sum() + torch.abs(self.G_core_imag).sum()
                for factor in [self.U_i, self.U_j, self.U_a, self.U_r, self.U_m]:
                    l1_loss += torch.abs(factor).sum()
        return l1_loss

    def helmholtz_residual(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared residual of (∇^2 + k^2) H_ij = 0
        coords: (B,2) with requires_grad=True
        """
        coords = self.coords.clone().requires_grad_(True)
        H = self.forward(coords)  # complex tensor (B,2,2)
        k2 = self.kq ** 2
        # accumulate squared residual
        res = 0.0
        for comp in (H.real, H.imag):
            # comp: (B,2,2) real
            B = comp.shape[0]
            # loop over tensor components
            for i in range(2):
                for j in range(2):
                    Hij = comp[:, i, j]  # shape (B,)
                    # first derivatives ∂H/∂x, ∂H/∂y
                    grads = torch.autograd.grad(
                        Hij.sum(), coords, create_graph=True
                    )[0]  # shape (B,2)
                    # build Laplacian
                    lap = 0.0
                    for d in range(2):
                        g_d = grads[:, d]  # (B,)
                        second = torch.autograd.grad(
                            g_d.sum(), coords, create_graph=True
                        )[0][:, d]  # (B,)
                        lap = lap + second
                    # residual = ∇²Hij + k² Hij
                    res = res + ((lap + k2 * Hij) ** 2).sum()
        # return mean over all coords & components
        return res / coords.numel()

    def double_curl_residual(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算 double curl PDE 残差损失: (∇×∇×G + k^2 G) = 0
        coords: (B,2) with requires_grad=True
        """
        coords = coords.clone().requires_grad_(True)
        G = self.forward(coords)  # (B, 2, 2)
        k2 = self.kq ** 2
        res = 0.0
        for comp in (G.real, G.imag):
            # 对每一列（即每个输出分量）分别做double curl
            for j in range(2):
                Gj = comp[:, :, j]  # (B, 2) 第j列
                # 1. divergence: div(Gj) = dGx/dx + dGy/dy
                div = 0.0
                for i in range(2):
                    grad = torch.autograd.grad(Gj[:, i].sum(), coords, create_graph=True)[0][:, i]
                    div = div + grad  # (B,)
                # 2. grad(div(Gj)): shape (B,2)
                grad_div = torch.stack([
                    torch.autograd.grad(div.sum(), coords, create_graph=True)[0][:, 0],  # d(div)/dx
                    torch.autograd.grad(div.sum(), coords, create_graph=True)[0][:, 1],  # d(div)/dy
                ], dim=1)  # (B,2)
                # 3. Laplacian of Gj: ΔGj_i = d²Gj_i/dx² + d²Gj_i/dy²
                lap = torch.zeros_like(Gj)
                for i in range(2):
                    grad_i = torch.autograd.grad(Gj[:, i].sum(), coords, create_graph=True)[0]
                    for d in range(2):
                        second = torch.autograd.grad(grad_i[:, d].sum(), coords, create_graph=True)[0][:, d]
                        lap[:, i] = lap[:, i] + second
                # 4. double curl: grad_div - lap
                double_curl = grad_div - lap  # (B,2)
                # 5. PDE残差: double_curl + k^2 * Gj
                pde_res = double_curl + k2 * Gj  # (B,2)
                res = res + (pde_res ** 2).sum()
        return res / coords.numel()

    def l2norm_loss(self,coords: torch.Tensor) -> torch.Tensor:
        coords = self.coords.clone().requires_grad_(True)
        G = self.forward(coords)
        return torch.sum(torch.sum(torch.real(G.conj() * G)))

    def divergence_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算无散度损失：div(G) = ∂G_{ij}/∂x_i，返回L2范数均值。
        coords: (B,2) with requires_grad=True
        """
        coords = coords.clone().requires_grad_(True)
        G = self.forward(coords)  # (B, 2, 2)
        B = G.shape[0]
        div = torch.zeros(B, 2, dtype=G.dtype, device=coords.device)  # (B, 2)
        for j in range(2):  # 对每一列
            # G_0j, G_1j 分别对x和y求偏导
            grads = torch.autograd.grad(G[:, 0, j].sum().real, coords, create_graph=True)[0][:, 0] \
                  + torch.autograd.grad(G[:, 1, j].sum().real, coords, create_graph=True)[0][:, 1] \
                  + torch.autograd.grad(G[:, 0, j].sum().imag, coords, create_graph=True)[0][:, 0] \
                  + torch.autograd.grad(G[:, 1, j].sum().imag, coords, create_graph=True)[0][:, 1]
            div[:, j] = grads
        # L2范数
        loss = (div.abs() ** 2).sum() / div.numel()
        return loss


import torch
import torch.nn as nn
from math import pi

class BesselFourier2x2(nn.Module):
    """
    Minimal, operator-aware Green's kernel (2D).

    Helmholtz (outgoing):
      - H0  term:  k^2 H0(kr) * I         with complex coeff cH0
      - H1  term: -(k/r) H1(kr) * I       with complex coeff cH1
      - H2  term:  k^2 H2(kr) * (n n^T)   with complex coeff cH2
      and the wavenumber k is LEARNABLE.

    Laplace (zero-frequency):
      - isotropic:  L0(r) = -(1/2π) log r        with complex coeff cL0
      - quadrupole: L2(r) =  (1/2π) r^{-2}
          we expose TWO learnable complex coeffs on the angular parts:
            * cL2c on cos(2θ)
            * cL2s on sin(2θ)
        i.e. we model only the anisotropic parts of n_i n_j:
            0.5*cos2θ * [[1,0],[0,-1]]  +  0.5*sin2θ * [[0,1],[1,0]]

    Learnable params (all small-normal init, non-zero):
      - logk  (k = softplus(logk) > 0)
      - cH0, cH1, cH2 (complex scalars via separate Re/Im tensors)
      - cL0, cL2c, cL2s (complex)
      - global complex scale (gs_re, gs_im)

    Methods kept: lasso_individual(), lasso_group_family(),
                  activity_family(), activity_individual(),
                  helmholtz_residual(), double_curl_residual(), etc.
    """

    def __init__(self,
                 kq: float = 1.0,
                 learn_Helmholtz: bool = True,
                 learn_Laplace: bool = True):
        super().__init__()
        self.learn_H = bool(learn_Helmholtz)
        self.learn_L = bool(learn_Laplace)

        # learnable wavenumber k (positive via softplus)
        self.logk = nn.Parameter(torch.tensor(float(kq)).log())

        # global complex scale
        self.gs_re = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gs_im = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # helper: complex scalar (re+im)
        def cparam():
            return nn.Parameter(torch.empty(1)), nn.Parameter(torch.empty(1))

        if self.learn_H:
            self.cH0_re, self.cH0_im = cparam()  # k^2 H0(kr) * I
            self.cH1_re, self.cH1_im = cparam()  # -(k/r) H1(kr) * I
            self.cH2_re, self.cH2_im = cparam()  # k^2 H2(kr) * (n n^T)

        if self.learn_L:
            self.cL0_re,  self.cL0_im  = cparam()  # L0 * I
            # separate cos2θ / sin2θ coefficients on L2:
            self.cL2c_re, self.cL2c_im = cparam()  # L2 * (0.5*cos2θ) * [[1,0],[0,-1]]
            self.cL2s_re, self.cL2s_im = cparam()  # L2 * (0.5*sin2θ) * [[0,1],[1,0]]

        self.reset_parameters(std=1e-2)

    # ---------- init (non-zero small normal) ----------
    def reset_parameters(self, std: float = 1e-2):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in ['gs_re', 'gs_im', 'logk']:
                    p.data.normal_(0.0, std)
                else:
                    p.data.normal_(0.0, std)

    # ---------- Hankel helpers ----------
    @staticmethod
    def _J(n, z):
        if n == 0: return torch.special.bessel_j0(z)
        if n == 1: return torch.special.bessel_j1(z)
        z = z + 1e-8
        Jm2, Jm1 = torch.special.bessel_j0(z), torch.special.bessel_j1(z)
        for k in range(2, n+1):
            Jk = (2*(k-1)/z)*Jm1 - Jm2
            Jm2, Jm1 = Jm1, Jk
        return Jm1

    @staticmethod
    def _Y(n, z):
        if n == 0: return torch.special.bessel_y0(z)
        if n == 1: return torch.special.bessel_y1(z)
        z = z + 1e-8
        Ym2, Ym1 = torch.special.bessel_y0(z), torch.special.bessel_y1(z)
        for k in range(2, n+1):
            Yk = (2*(k-1)/z)*Ym1 - Ym2
            Ym2, Ym1 = Ym1, Yk
        return Ym1

    @staticmethod
    def _H1(n, z):
        return BesselFourier2x2._J(n, z) + 1j * BesselFourier2x2._Y(n, z)

    # ---------- forward ----------
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B,2) float
        returns: (B,2,2) complex
        """
        y, x = coords.unbind(-1)
        r = torch.sqrt(x*x + y*y + 1e-12)
        th = torch.atan2(y, x)
        rinv = 1.0 / (r + 1e-12)
        cx, cy = x*rinv, y*rinv
        cos2t = cx*cx - cy*cy
        sin2t = 2.0*cx*cy

        B = coords.shape[0]
        out = torch.zeros(B, 2, 2, dtype=torch.complex64, device=coords.device)

        # ---------- Helmholtz ----------
        if self.learn_H:
            # learnable k
            k = torch.nn.functional.softplus(self.logk) + 1e-8
            kr = k * r

            H0 = self._H1(0, kr)
            H1 = self._H1(1, kr)
            # H2 via recurrence: J2 = 2 J1/x - J0  (and same for Y)
            J2 = 2*torch.special.bessel_j1(kr)/(kr + 1e-12) - torch.special.bessel_j0(kr)
            Y2 = 2*torch.special.bessel_y1(kr)/(kr + 1e-12) - torch.special.bessel_y0(kr)
            H2 = J2 + 1j*Y2

            cH0 = (self.cH0_re + 1j*self.cH0_im).to(H0.dtype)
            cH1 = (self.cH1_re + 1j*self.cH1_im).to(H0.dtype)
            cH2 = (self.cH2_re + 1j*self.cH2_im).to(H0.dtype)

            iso  = (k*k) * H0 * cH0 + (-(k*rinv) * H1) * cH1   # (B,)
            quad = (k*k) * H2 * cH2                            # (B,)

            out[:,0,0] += iso + quad * (cx*cx)
            out[:,1,1] += iso + quad * (cy*cy)
            off = quad * (cx*cy)
            out[:,0,1] += off
            out[:,1,0] += off

        # ---------- Laplace ----------
        if self.learn_L:
            r_safe = r + 1e-12
            L0 = -(1.0/(2*pi)) * torch.log(r_safe)     # isotropic scalar
            L2 =  (1.0/(2*pi)) * r_safe.pow(-2.0)      # quadrupole radial scalar

            cL0  = (self.cL0_re  + 1j*self.cL0_im ).to(out.dtype)
            cL2c = (self.cL2c_re + 1j*self.cL2c_im).to(out.dtype)  # cos2θ coeff
            cL2s = (self.cL2s_re + 1j*self.cL2s_im).to(out.dtype)  # sin2θ coeff

            isoL   = L0 * cL0
            # anisotropic tensor from cos2θ / sin2θ pieces:
            # 0.5*cos2θ * [[ 1, 0],[ 0,-1]] + 0.5*sin2θ * [[0,1],[1,0]]
            a = 0.5 * L2 * cL2c * cos2t
            b = 0.5 * L2 * cL2s * sin2t

            out[:,0,0] += isoL + a
            out[:,1,1] += isoL - a
            out[:,0,1] += b
            out[:,1,0] += b

        gamma = (self.gs_re + 1j*self.gs_im).to(out.dtype)
        return out * gamma

    # ---------- regularizers ----------
    def lasso_individual(self):
        """L1 over individual complex coefficients (sum |Re|+|Im|)."""
        l1 = 0.0
        # include |logk| to mildly regularize k if you like (optional, commented):
        # l1 = l1 + self.logk.abs()

        if self.learn_H:
            l1 = (l1 + self.cH0_re.abs() + self.cH0_im.abs()
                      + self.cH1_re.abs() + self.cH1_im.abs()
                      + self.cH2_re.abs() + self.cH2_im.abs())
        if self.learn_L:
            l1 = (l1 + self.cL0_re.abs()  + self.cL0_im.abs()
                      + self.cL2c_re.abs() + self.cL2c_im.abs()
                      + self.cL2s_re.abs() + self.cL2s_im.abs())
        return l1

    def lasso_group_family(self):
        """Group lasso: L2 (or L1 if you prefer) over families."""
        norms = []
        if self.learn_H:
            v = torch.stack([
                self.cH0_re, self.cH0_im,
                self.cH1_re, self.cH1_im,
                self.cH2_re, self.cH2_im
            ])
            norms.append(torch.linalg.vector_norm(v, ord=2))
        if self.learn_L:
            v = torch.stack([
                self.cL0_re,  self.cL0_im,
                self.cL2c_re, self.cL2c_im,
                self.cL2s_re, self.cL2s_im
            ])
            norms.append(torch.linalg.vector_norm(v, ord=2))
        return sum(norms)

    # ---------- activity (who's hot) ----------
    @torch.no_grad()
    def activity_family(self):
        d = {}
        k = float(torch.nn.functional.softplus(self.logk).cpu())
        d['k_learned'] = k
        if self.learn_H:
            d['Helmholtz_H0_re'] = float(self.cH0_re.cpu())
            d['Helmholtz_H0_im'] = float(self.cH0_im.cpu())
            d['Helmholtz_H1_re'] = float(self.cH1_re.cpu())
            d['Helmholtz_H1_im'] = float(self.cH1_im.cpu())
            d['Helmholtz_H2_re'] = float(self.cH2_re.cpu())
            d['Helmholtz_H2_im'] = float(self.cH2_im.cpu())
        if self.learn_L:
            d['Laplace_iso_re']   = float(self.cL0_re.cpu())
            d['Laplace_iso_im']   = float(self.cL0_im.cpu())
            d['Laplace_cos2_re']  = float(self.cL2c_re.cpu())
            d['Laplace_cos2_im']  = float(self.cL2c_im.cpu())
            d['Laplace_sin2_re']  = float(self.cL2s_re.cpu())
            d['Laplace_sin2_im']  = float(self.cL2s_im.cpu())
        return d

    @torch.no_grad()
    def activity_individual(self):
        return self.activity_family()

    # ---------- physics losses (kept) ----------
    def helmholtz_residual(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        G = self.forward(coords)
        k = torch.nn.functional.softplus(self.logk) + 1e-8
        k2 = k**2
        res = 0.0
        for comp in (G.real, G.imag):
            for i in range(2):
                for j in range(2):
                    Hij = comp[:, i, j]
                    grad = torch.autograd.grad(Hij.sum(), coords, create_graph=True)[0]
                    lap = 0.0
                    for d in range(2):
                        second = torch.autograd.grad(grad[:, d].sum(), coords, create_graph=True)[0][:, d]
                        lap = lap + second
                    res = res + ((lap + k2*Hij)**2).mean()
        return res

    def double_curl_residual(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        G = self.forward(coords)
        k = torch.nn.functional.softplus(self.logk) + 1e-8
        k2 = k**2
        res = 0.0
        for comp in (G.real, G.imag):
            for j in range(2):
                Gj = comp[:, :, j]
                dGx_dx = torch.autograd.grad(Gj[:,0].sum(), coords, create_graph=True)[0][:,0]
                dGy_dy = torch.autograd.grad(Gj[:,1].sum(), coords, create_graph=True)[0][:,1]
                div = dGx_dx + dGy_dy
                gdiv = torch.autograd.grad(div.sum(), coords, create_graph=True)[0]
                lap = torch.zeros_like(Gj)
                for i in range(2):
                    gi = torch.autograd.grad(Gj[:,i].sum(), coords, create_graph=True)[0]
                    lap[:,i] = (torch.autograd.grad(gi[:,0].sum(), coords, create_graph=True)[0][:,0] +
                                torch.autograd.grad(gi[:,1].sum(), coords, create_graph=True)[0][:,1])
                res = res + ((gdiv - lap + k2*Gj)**2).mean()
        return res

    def l2norm_loss(self, coords: torch.Tensor) -> torch.Tensor:
        G = self.forward(coords)
        return torch.real((G.conj()*G).sum()) / coords.shape[0]

    def divergence_loss(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        G = self.forward(coords)
        div_norm = 0.0
        for j in range(2):
            for part in (G.real, G.imag):
                d0dx = torch.autograd.grad(part[:,0,j].sum(), coords, create_graph=True)[0][:,0]
                d1dy = torch.autograd.grad(part[:,1,j].sum(), coords, create_graph=True)[0][:,1]
                div = d0dx + d1dy
                div_norm += (div**2).mean()
        return div_norm
