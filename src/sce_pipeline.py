import torch
from helper.npcf_calculation import *
from helper.utils import *

class EffectiveConductivityNPCF:
    """
    A refactored class for computing NPCF-based effective conductivity with optional
    intermediate caching and/or custom inputs for partial computations.
    """

    def __init__(self,
                 size,
                 sigma0,
                 sigma1,
                 n,
                 d,
                 device,
                 gt_size = None):
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.size = size
        self.n = n  # up to which order you want to compute
        self.d = d  # dimension, presumably 2 in your use-case
        self.device = device
        self.gt_size = size # Fixed as size to avoid scaling issue
        self.radius_scale = self.gt_size / self.size

        # Initialize placeholders for microstructure and correlation data
        self.microstructure = None
        self.S2 = None
        self.S3 = None
        self.S4 = None
        self.phi = None
        self.A2 = None
        self.A3 = None
        self.A4 = None
        self.D = None
        self.Sigma = None

        self.normalization_factor = 1


    # -------------------------------------------------------------------------
    #                 Basic Utilities
    # -------------------------------------------------------------------------

    def compute_T(self, r: torch.Tensor, d: int) -> torch.Tensor:
        r_norm = torch.norm(r, dim=1, keepdim=True).clamp_min(1e-8)
        n = r / r_norm
        # Calculate T for all r using broadcasting
        T = (
                (d * torch.einsum('ij,ik->ijk', n, n) -
                 torch.eye(d, device=r.device).unsqueeze(0))
                / (r_norm.unsqueeze(2) * self.radius_scale) ** d
        )
        return T

    def compute_T_with_NN(self, r, model):
        return model(r)

    def compute_beta(self) -> float:
        return (self.sigma1 - self.sigma0) / (self.sigma1 + (self.d - 1) * self.sigma0)

    def compute_D_target(self, gt, beta = None, phi = None) -> torch.Tensor:
        if beta is None:
            beta = self.beta
            if beta is None:
                raise ValueError("beta not provided or set in class.")
        if phi is None:
            phi = self.phi
            if phi is None:
                raise ValueError("phi not provided or set in class.")
        d = self.d  # Dimensionality of the matrix
        sigma0 = self.sigma0
        I = torch.eye(d, device=gt.device)  # Identity matrix

        # Compute the inverse term
        Sigma_e_minus_sigma0I_inv = torch.linalg.inv(gt - sigma0 * I)

        # Compute the main expression
        D_target = (beta ** 2 * phi ** 2) * Sigma_e_minus_sigma0I_inv @ ((d - 1) * sigma0 * I + gt)

        return D_target


    # -------------------------------------------------------------------------
    #                Correlation Functions
    # -------------------------------------------------------------------------
    def compute_phi(self, microstructure=None, store=True) -> float:
        """
        Compute the volume fraction phi (the mean of the microstructure).
        If `microstructure` is None, use `self.microstructure`.
        If `store=True`, store result in self.phi.
        """
        if microstructure is None:
            microstructure = self.microstructure
            if microstructure is None:
                raise ValueError("microstructure not provided or set in class.")
        phi = torch.mean(microstructure)
        if store:
            self.phi = phi
        return phi

    def compute_S2(self, microstructure=None, store=True) -> torch.Tensor:
        """
        Compute the two-point correlation function S2.
        If `microstructure` is None, we use `self.microstructure`.
        If `store=True`, store the result in self.S2.
        """
        if microstructure is None:
            microstructure = self.microstructure
            if microstructure is None:
                raise ValueError("microstructure not provided or set in class.")
        # This function calls your 'twopcf'
        S2 = twopcf(microstructure)
        if store:
            self.S2 = S2
        return S2

    def compute_S3(self, microstructure=None, store=True) -> torch.Tensor:
        """
        Compute the three-point correlation S3 using `threepcf_fullset`.
        """
        if microstructure is None:
            microstructure = self.microstructure
            if microstructure is None:
                raise ValueError("microstructure not provided or set in class.")
        S3 = threepcf_fullset(microstructure)
        if store:
            self.S3 = S3
        return S3

    def compute_S4(self, microstructure=None, store=True) -> torch.Tensor:
        """
        Compute the four-point correlation S4 using `fourpcf_fullset`.
        """
        if microstructure is None:
            microstructure = self.microstructure
            if microstructure is None:
                raise ValueError("microstructure not provided or set in class.")
        S4 = fourpcf_fullset(microstructure)
        if store:
            self.S4 = S4
        return S4

    # -------------------------------------------------------------------------
    #                A2, A3, A4 Calculation
    # -------------------------------------------------------------------------

    def compute_A2(self,
                   beta: float = None,
                   phi: float = None,
                   S2: torch.Tensor = None,
                   T_marix: torch.Tensor = None,
                   use_NN=False,
                   model=None,
                   delta=None,
                   store=True,get_T=False,get_T_and_A=False) -> torch.Tensor:
        """
        Compute the A2 term, with possible custom S2, phi, etc.
        If none given, fallback to self.S2, self.phi, etc.
        If store=True, save the result to self.A2.
        """
        d = 2
        omega = 2 * torch.pi
        n = 2


        I = J = self.size
        a = torch.repeat_interleave(torch.arange(I), I).view(-1, 1).to(self.device)
        b = torch.arange(J).repeat(J).view(-1, 1).to(self.device)
        indices = torch.cat((b, a), dim=1).to(self.device) # xy vs row col
        indices -= int(self.size / 2)
        norms = torch.norm(indices.float(), dim=1)

        valid_mask = norms != 0
        valid_indices = indices[valid_mask]

        # valid_indices_ = valid_indices.float().requires_grad_(True)

        valid_indices_ = torch.as_tensor(valid_indices, dtype=torch.float32).requires_grad_(True)
        # T for all valid indices
        if T_marix is not None:
            T_matrices = T_marix
        else:
            if not use_NN:
                T_matrices = self.compute_T(valid_indices_, d)
            else:
                if model is None:
                    raise ValueError("use_NN=True but no model provided.")
                # From your snippet you had: T_matrices = self.compute_T_with_NN(valid_indices.float(), model) / 70056.7969
                T_matrices = self.compute_T_with_NN(valid_indices_, model).real / self.normalization_factor
                # loss = get_mixed_partial_loss(T_matrices, valid_indices_)
            if get_T:
                return T_matrices


        if S2 is None:
            if self.S2 is None:
                raise ValueError("S2 not provided and self.S2 is None. Please compute or provide S2.")
            S2 = self.S2
        # Pull from class if not provided
        if beta is None:
            beta = self.compute_beta()
        if phi is None:
            if self.phi is None:
                raise ValueError("phi not provided and self.phi is None. Please compute or provide phi.")
            phi = self.phi


        # Reshape S2
        S = S2.to(self.device)
        valid_S = S.flatten()[valid_mask]

        # Delta
        if delta is None:
            Delta_matrices = (torch.abs(valid_S) - phi ** 2).to(self.device)
        else:
            Delta_matrices = delta.to(self.device)
        A = torch.einsum('ijk,i->jk', T_matrices, Delta_matrices.float())

        # scalar multipliers
        A *= (self.radius_scale ** d) ** (n - 1) * d / omega

        if store:
            self.A2 = A
        if get_T_and_A:
            return A, T_matrices, valid_indices_
        return A


    def compute_A3(
            self,
            beta: float = None,
            phi: float = None,
            S2: torch.Tensor = None,
            S3: torch.Tensor = None,
            use_NN: bool = False,
            model=None,
            delta: torch.Tensor = None,
            mask: torch.Tensor = None,  # <--- newly added
            store: bool = True,
            masked_data: bool = True,
            get_T: bool = False,
            T1_input = None,
            T2_input = None,
    ) -> torch.Tensor:
        """
        Compute the A3 term. If a mask is provided, only compute over the
        subset of S3 indices that are True in the mask. Otherwise, compute
        over the entire S3.
        """
        d = 2
        n = 3
        omega = 2 * torch.pi
        phi = float(phi) if phi is not None else None


        if mask is not None:
            if masked_data:
                # 1) Create zeros of shape [64^4]
                s3_unmasked = torch.zeros(64 ** 4, device=self.device).to(torch.float32)
                mask = mask.to(self.device)
                # 2) Scatter masked values in
                s3_unmasked[mask] = S3  # mask is True on positions we want

                # 3) Reshape to [64,64,64,64]
                S3 = s3_unmasked.view(64, 64, 64, 64)


        I = J = self.size
        I3 = J3 = K3 = L3 = self.size

        # Build the full coordinate grid [64^4, 4]
        # (assuming self.size == 64, or that I3=J3=K3=L3=self.size)
        indices_all = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(I3),
                        torch.arange(J3),
                        torch.arange(K3),
                        torch.arange(L3),
                    ),
                    dim=-1
                )
                - int(self.size // 2)
        ).reshape(-1, 4).to(self.device)  # shape [I3*J3*K3*L3, 4]

        if mask is None:
            # ==============================================
            # ORIGINAL LOGIC (use entire S3)
            # ==============================================
            indices = indices_all
        else:
            # ==============================================
            # FILTER TO ONLY THOSE ENTRIES WHERE mask==True
            # ==============================================
            # mask is assumed to be shape [I3*J3*K3*L3], bool
            if mask.shape[0] != I3 * J3 * K3 * L3:
                raise ValueError(f"Mask must be shape [{I3 * J3 * K3 * L3}], got {mask.shape[0]} instead.")

            # Subset the big index array by this mask
            indices = indices_all[mask]

        # Now indices is shape [M,4] for either M=I3*J3*K3*L3 (no mask) or M <= that (masked)

        # Extract r1, r2 from the subset indices
        # r1 = indices[:, :2].float()  # shape [M,2]
        # r2 = indices[:, 2:].float()  # shape [M,2]

        r1 = indices[:, [1, 0]].float() # make it yx instead of xy
        r2 = indices[:, [3, 2]].float()

        # Identify valid (non-zero) positions
        valid_r1 = (r1.norm(dim=1) != 0)
        valid_r2 = (r2.norm(dim=1) != 0)
        valid_mask = valid_r1 & valid_r2

        # Subset further for valid
        indices_valid = indices[valid_mask]
        r1_valid = r1[valid_mask]
        r2_valid = r2[valid_mask]
        M_valid = indices_valid.shape[0]

        # Compute T1 and T2 (optionally via NN)
        if T1_input is not None and T2_input is not None:
            T1 = T1_input
            T2 = T2_input
        else:
            if not use_NN:
                T1 = self.compute_T(r1_valid, d)  # shape [M_valid, d, something]
                T2 = self.compute_T(r2_valid, d)  # shape [M_valid, d, something]
            else:
                if model is None:
                    raise ValueError("use_NN=True but no model provided.")
                T1 = self.compute_T_with_NN(r1_valid, model) / self.normalization_factor
                T2 = self.compute_T_with_NN(r2_valid, model) / self.normalization_factor
            if get_T:
                return T1, T2



        # Pull from class if not provided
        if beta is None:
            beta = self.compute_beta()
        if phi is None:
            if self.phi is None:
                raise ValueError("phi not provided and self.phi is None. Please compute or provide phi.")
            phi = self.phi
        if S2 is None:
            if self.S2 is None:
                raise ValueError("S2 not provided and self.S2 is None. Please compute or provide S2.")
            S2 = self.S2
        if S3 is None:
            if self.S3 is None:
                raise ValueError("S3 not provided and self.S3 is None. Please compute or provide S3.")
            S3 = self.S3

        # Move to device and do same permutations as in original code
        S2 = S2.to(self.device)
        S3 = S3.to(self.device).to(torch.float32)

        # Construct or retrieve Delta
        if delta is None:
            # S3-based determinant
            # shape for each row: 2x2 matrix => det( [ [S2(r1), phi], [S3(4D index), S2(r2)] ] )
            Delta_subset = torch.linalg.det(
                torch.stack([
                    torch.stack([
                        S2[r1_valid[:, 0].long(), r1_valid[:, 1].long()],
                        torch.full((M_valid,), phi, device=self.device)
                    ], dim=1),
                    torch.stack([
                        S3[
                            indices_valid[:, 0].long(),
                            indices_valid[:, 1].long(),
                            indices_valid[:, 2].long(),
                            indices_valid[:, 3].long()
                        ],
                        S2[r2_valid[:, 0].long(), r2_valid[:, 1].long()]
                    ], dim=1),
                ], dim=2)
            )
        else:
            # Use provided delta, but presumably also subset
            # (Make sure delta is shaped [I3*J3*K3*L3], then subset)
            if delta.shape[0] != I3 * J3 * K3 * L3:
                raise ValueError("delta must be size [I3*J3*K3*L3].")
            delta_valid = delta[mask] if mask is not None else delta
            Delta_subset = delta_valid[valid_mask]

        Delta_subset = Delta_subset.float()

        # T1, T2 have shape [M_valid, d, d?], we multiply T1*T2 => T_prod has shape [M_valid, d, d]
        T_prod = torch.einsum('ijk,ikl->ijl', T1, T2)

        # Weighted sum by Delta over M_valid => shape [d, d]
        A_subset = torch.einsum('ijk,i->jk', T_prod, Delta_subset)

        # Apply prefactor
        # (self.radius_scale ** d)^(n-1) * (-1.0/phi)^(n-2) * (d/omega)^(n-1)
        A_subset = (self.radius_scale ** d) ** (n - 1) \
                   * (-1.0 / phi) ** (n - 2) \
                   * (d / omega) ** (n - 1) \
                   * A_subset

        # Store if desired
        if store:
            self.A3 = A_subset

        return A_subset

    def compute_A4(self,
                   beta: float = None,
                   phi: float = None,
                   S2: torch.Tensor = None,
                   S3: torch.Tensor = None,
                   S4: torch.Tensor = None,
                   store=True) -> torch.Tensor:
        """
        Compute A4 from your snippet.
        """
        d = 2
        n = 4
        omega = 2 * torch.pi

        if beta is None:
            beta = self.compute_beta()
        if phi is None:
            if self.phi is None:
                raise ValueError("phi not provided and self.phi is None. Please compute or provide phi.")
            phi = self.phi
        if S2 is None:
            S2 = self.S2
        if S3 is None:
            S3 = self.S3
        if S4 is None:
            S4 = self.S4

        # Re-permute
        S2 = S2
        S3 = S3
        S4 = S4

        (J, I) = S2.shape
        (J4, I4, L4, K4, O4, P4) = S4.shape

        indices = (
                torch.stack(torch.meshgrid(torch.arange(I4),
                                           torch.arange(J4),
                                           torch.arange(K4),
                                           torch.arange(L4),
                                           torch.arange(O4),
                                           torch.arange(P4)), dim=-1)
                - int(self.size / 2)
        ).reshape(-1, 6).to(self.device)

        r1 = indices[:, :2].float()
        r2 = indices[:, 2:4].float()
        r3 = indices[:, 4:].float()

        valid_r1 = torch.norm(r1, dim=1) != 0
        valid_r2 = torch.norm(r2, dim=1) != 0
        valid_r3 = torch.norm(r3, dim=1) != 0

        T1 = torch.where(valid_r1[:, None, None], self.compute_T(r1, d), torch.eye(d, device=self.device))
        T2 = torch.where(valid_r2[:, None, None], self.compute_T(r2, d), torch.eye(d, device=self.device))
        T3 = torch.where(valid_r3[:, None, None], self.compute_T(r3, d), torch.eye(d, device=self.device))

        valid_mask = valid_r1 & valid_r2 & valid_r3

        # Delta (vectorized)
        # This block is from your snippet. Careful about the shape.
        Delta = torch.det(
            torch.stack([
                torch.stack([
                    S2[r1[:, 0].long(), r1[:, 1].long()],
                    torch.full((indices.shape[0],), phi, device=self.device),
                    torch.zeros(indices.shape[0], device=self.device)
                ], dim=1),
                torch.stack([
                    S3[r1[:, 0].long(), r1[:, 1].long(),
                    r2[:, 0].long(), r2[:, 1].long()],
                    S2[r2[:, 0].long(), r2[:, 1].long()],
                    torch.full((indices.shape[0],), phi, device=self.device)
                ], dim=1),
                torch.stack([
                    S4[indices[:, 0].long(), indices[:, 1].long(),
                    indices[:, 2].long(), indices[:, 3].long(),
                    indices[:, 4].long(), indices[:, 5].long()],
                    S3[r2[:, 0].long(), r2[:, 1].long(),
                    r3[:, 0].long(), r3[:, 1].long()],
                    S2[r3[:, 0].long(), r3[:, 1].long()]
                ], dim=1)
            ])
        )

        T12 = torch.einsum('ijk,ikl->ijl', T1[valid_mask], T2[valid_mask])
        T_prod = torch.einsum('ijk,ikl->ijl', T12, T3[valid_mask])
        A = torch.einsum('ijk,i->jk', T_prod, Delta[valid_mask])

        A *= (self.radius_scale ** d) ** (n - 1) * (-1.0 / phi) ** (n - 2) * (d / omega) ** (n - 1)
        if store:
            self.A4 = A
        return A

    # -------------------------------------------------------------------------
    #                D and Sigma Calculation
    # -------------------------------------------------------------------------
    def compute_D(self,
                  phi: float = None,
                  A2: torch.Tensor = None,
                  A3: torch.Tensor = None,
                  A4: torch.Tensor = None,
                  store=True) -> torch.Tensor:
        """
        Compute D = beta * phi * I - A2 * beta^2 - A3 * beta^3 - A4 * beta^4
        up to the order n that you specify in self.n.
        If store=True, store result in self.D.
        """
        if phi is None:
            phi = self.phi
        if phi is None:
            raise ValueError("phi is None, cannot compute D. Provide or compute phi first.")

        beta = self.compute_beta()
        d = self.d
        I = torch.eye(d, device=self.device)

        # Gather needed A2..A4 up to n
        if A2 is None and self.n >= 2:
            if self.A2 is None:
                raise ValueError("A2 not computed or provided.")
            A2 = self.A2
        if A3 is None and self.n >= 3:
            if self.A3 is None:
                raise ValueError("A3 not computed or provided.")
            A3 = self.A3
        if A4 is None and self.n >= 4:
            if self.A4 is None:
                raise ValueError("A4 not computed or provided.")
            A4 = self.A4
        D = beta * phi * I
        if self.n >= 2 and A2 is not None:
            D -= A2 * (beta ** 2)
        if self.n >= 3 and A3 is not None:
            D -= A3 * (beta ** 3)
        if self.n >= 4 and A4 is not None:
            D -= A4 * (beta ** 4)

        if store:
            self.D = D
        return D

    def compute_Sigma(self, D=None, phi=None, store=True) -> torch.Tensor:
        """
        Compute final Sigma from D:

        Sigma = inverse( (D / (beta^2 (phi+eps)^2)) - I ) * [ (sigma0 / (beta^2 (phi+eps)^2)) * D + (d-1)*sigma0*I ]

        If store=True, store result in self.Sigma.
        """
        eps = 1e-12
        d = self.d
        if D is None:
            D = self.D
        if D is None:
            raise ValueError("D is None, cannot compute Sigma. Provide or compute D first.")

        if phi is None:
            phi = self.phi
        if phi is None:
            raise ValueError("phi is None, cannot compute Sigma. Provide or compute phi first.")

        beta = self.compute_beta()
        I = torch.eye(d, device=self.device)

        # The final formula
        Sigma = torch.inverse(D / beta ** 2 / (phi + eps) ** 2 - torch.eye(d, device=self.device)) \
                @ (self.sigma0 / beta ** 2 / (phi + eps) ** 2 * D + (d - 1) * self.sigma0 * torch.eye(d,
                                                                                                      device=self.device))
        # Sigma2 = torch.inverse(D / beta ** 2 / (phi + eps) ** 2 - torch.eye(d, device=self.device)) \
        #         * (self.sigma0 / beta ** 2 / (phi + eps) ** 2 * D + (d - 1) * self.sigma0 * torch.eye(d,
        #                                                                                               device=self.device))
        if store:
            self.Sigma = Sigma
        return Sigma

    # -------------------------------------------------------------------------
    #                Full Pipeline
    # -------------------------------------------------------------------------
    def compute_full(self, microstructure=None):
        """
        High-level pipeline that:
        1. sets self.microstructure from input
        2. computes phi, S2, S3, S4 up to self.n
        3. computes A2, A3, A4 up to self.n
        4. computes D
        5. computes Sigma
        6. returns Sigma
        """
        # 1) set microstructure if given
        if microstructure is not None:
            self.microstructure = microstructure

        if self.microstructure is None:
            raise ValueError("No microstructure found or provided.")

        # 2) compute phi, S2, S3, S4
        self.compute_phi(store=True)  # self.phi
        if self.n >= 2:
            self.compute_S2(store=True)  # self.S2
        if self.n >= 3:
            self.compute_S3(store=True)  # self.S3
        if self.n >= 4:
            self.compute_S4(store=True)  # self.S4

        # 3) compute A2, A3, A4
        if self.n >= 2:
            self.compute_A2(store=True)
        if self.n >= 3:
            self.compute_A3(store=True)
        if self.n >= 4:
            self.compute_A4(store=True)

        # 4) compute D
        self.compute_D(store=True)
        # 5) compute Sigma
        self.compute_Sigma(store=True)

        return self.Sigma



# -----------------------------------------------------------------
# Example usage and debugging
# -----------------------------------------------------------------

if __name__ == "__main__":
    from src.helper.microstructure_generation import *
    from helper.utils import *
    from pde import *

    config = load_config('./config.yaml')
    # sigma0 = config['sigma0']
    # sigma1 = config['sigma1']
    sigma0 = 5
    sigma1 = 20
    size = config['size']
    d = config['d']
    n = config['n']
    device_id = config['device']
    npcf = config['n']
    seed = config['seed']
    length_scale_x = [0.001]
    length_scale_y = [0.11]
    mean = [0.7]
    # gt_size = config['gt_size']
    gt_size = 6

    device = torch.device(f"cuda:{device_id}")

    microstructure_gt = get_microstructure(mode='generate', size=size, mean=mean[0], length_scale_x=length_scale_x[0],
                                           length_scale_y=length_scale_y[0], seed=seed)

    microstructure_gt_switch = 1-microstructure_gt

    phi, S2, S3 = get_patch_npcf(microstructure_gt,threepcf=True)
    phi_switch, S2_switch, S3_switch = get_patch_npcf(microstructure_gt_switch,threepcf=True)


    microstructure_gt = torch.as_tensor(microstructure_gt, device=device)

    sce_pipeline = EffectiveConductivityNPCF(
        size,sigma0, sigma1, n, d, device,gt_size)

    # Optionally set or confirm parameters directly on the instance if needed
    # npcf_instance_2.microstructure = microstructure_gt  # If your class expects to store microstructure

    # Compute the conductivity using .compute(), which uses up to n=2 if your code is set up that way
    # Sigma_2 = sce_pipeline.compute_full()
    # print("Computed 2nd-order conductivity:\n", Sigma_2)

    beta = sce_pipeline.compute_beta()

    # Debug: let's directly compute A2 again (you've done that above, but just to show usage)
    A2_debug = sce_pipeline.compute_A2(beta, phi, S2)
    A2_debug_2 = sce_pipeline.compute_A2(-beta, 1-phi, S2_switch)
    assert (torch.abs(A2_debug - A2_debug_2).mean()) == 0
    print("A2 (debug) shape:", A2_debug.shape)
    print("A2 (debug) sample values:", A2_debug.flatten()[:10])  # Print first 10 entries as an example

    # Similarly, try a 3rd-order instance for debugging if you like
    sce_pipeline_2 = EffectiveConductivityNPCF(
        size,sigma0, sigma1, 3, d, device,gt_size)

    sce_pipeline_2_switch = EffectiveConductivityNPCF(
        size,sigma1, sigma0, 3, d, device,gt_size)
    beta_switch = sce_pipeline_2_switch.compute_beta()


    # Debug: Compare A3 from the instance to your manual A3_1. Should be the same if inputs match.
    A3_debug = sce_pipeline_2.compute_A3(beta, phi,S2,S3)
    A3_debug_1 = sce_pipeline_2.compute_A3(beta_switch, phi_switch,S2_switch,S3_switch)

    D = sce_pipeline_2.compute_D(phi, A2_debug, A3_debug)
    D_2 = sce_pipeline.compute_D(phi, A2_debug, None)
    Sigma_2 = sce_pipeline.compute_Sigma(D_2, phi)
    Sigma_3 = sce_pipeline_2.compute_Sigma(D, phi)
    microstructure_gt = get_microstructure(mode='generate', size=1024, mean=mean[0], length_scale_x=length_scale_x[0],
                                           length_scale_y=length_scale_y[0], seed=seed)

    pde = effective_conductivity_pde(microstructure_gt, sigma0, sigma1).compute()
    print("Computed 2rd-order conductivity:\n", Sigma_2)
    print("Computed 3rd-order conductivity:\n", Sigma_3)
    print("Computed pde conductivity:\n", pde)
    print("A3 (debug) shape:", A3_debug.shape)
    # Check difference with your previously computed A3_1
    diff_A3 = torch.abs(A3_debug - A3_debug_1).mean()
    print("Mean absolute difference between A3 (debug) and A3_1:", diff_A3.item())

    # Additional debug checks
    # Example: difference between your final Sigma_3 and some reference if you have one
    # print("Difference vs reference Sigma_3:", torch.abs(Sigma_3 - some_reference_tensor).mean())

    print("Debugging complete!")
