from helper.utils import *
from sce_pipeline import *
from tqdm import tqdm
import torch.optim as optim
from model.bessel_fourier_wave import *
from model.fourier_conductivity import *

def normalize(input):
    normalized = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
    return normalized

def get_AB(D, beta, phi, sigma0, device='cpu'):
    """
    Compute the derivative of Sigma with respect to D in PyTorch.

    Parameters:
    - D (torch.Tensor): The input matrix D (assumed to be square and invertible).
    - beta (float): Scalar parameter beta.
    - phi (float): Scalar parameter phi.
    - sigma0 (float): Scalar parameter sigma0.
    - device (str): Device to perform computations ('cpu' or 'cuda').

    Returns:
    - dSigma_dD (torch.Tensor): The derivative of Sigma with respect to D.
    """
    d = D.shape[0]  # Dimensionality
    identity = torch.eye(d, device=device)  # Identity matrix

    # Compute A and B
    A = D / beta**2 / phi**2 - identity
    B = (sigma0 / beta**2 / phi**2) * D + (d - 1) * sigma0 * identity

    # Compute A inverse
    A_inv = torch.inverse(A)


    return A,B,A_inv
# load configs
config = load_config('config.yaml')
sigma0 = config['sigma0']
sigma1 = config['sigma1']
size = config['size']
d = config['d']
device_id = config['device']
n = config['n']
gt_size = config['gt_size']
device = torch.device(f"cuda:{device_id}")
mean = config['mean']
seed = config['seed']
realization_per_field = 30

def get_T(s, model=None):


    # Create indices array and norms
    I = J = s
    a = torch.repeat_interleave(torch.arange(I), I).view(-1, 1).to(
        device)  # Create the first column, repeating each number I times
    b = torch.arange(J).repeat(J).view(-1, 1).to(
        device)  # Create the second column, tiling the reverse sequence
    indices = torch.cat((a, b), dim=1).to(device)  # Combine the two columns horizontally
    indices -= int(size / 2)

    # Compute norms of indices
    norms = torch.norm(indices.float(), dim=1)

    # Filter out zero norms and the index [0, 0] if necessary
    valid_mask = norms != 0
    valid_indices = indices[valid_mask]

    T_matrices = model(valid_indices.float())

    return T_matrices

model_id = 'bessel'
model_ckpt = torch.load(f'./kernel_training_result_109_new/kernel_learning_2pcf_siren_v0_0.pt', map_location=device,
    weights_only=False,)
# model = FourierExpansion2x2(N=2).to(device)
model = BesselFourier2x2().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# if model_id == "toy":
#     model = SIREN_toy()
#     optimizer = optim.Adam(model.parameters(), lr=0.5)
#
# elif model_id == "siren":
#     # model = SymmetricMatrixSIREN(hidden_dim=512, hidden_layers=5, s=64).to(device)
#     model = SymmetricMatrixSIREN(hidden_dim=256, hidden_layers=3, s=64).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
# elif model_id == "fourier":
#     model = FourierExpansion2x2(N=2).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.1)
# elif model_id == "bassel":
#     model = BesselFourier2x2().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.1)
# else:
#     raise ValueError("model_id must be 'toy' or 'siren' or 'fourier'")
model.load_state_dict(model_ckpt.state_dict())

sce_pipeline = EffectiveConductivityNPCF(
    size, sigma0, sigma1, 2, d, device, gt_size)
# length_x = np.array([0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7])
# length_y = np.array([0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7])
length_x = np.array([0.001])
length_y = np.array([0.01])
num = 1
s = 64
Num = length_x.size*length_y.size
# new_T = torch.load('./total_T_v2_wo_init.pt')
# new_T = torch.zeros_like(new_T)
theta = np.arange(0, 361, 3)  # get theta in degree
beta = sce_pipeline.compute_beta()
# new_T = get_T(s,model).detach()
# new_T = sce_pipeline.compute_A2(beta,None,None,get_T=True)
new_T = sce_pipeline.compute_A2(beta=None, phi=None, S2=None, model=model, use_NN=True, get_T=True).detach()
# T1,T2 = sce_pipeline.compute_A3(beta=None, phi=None, S2=None, model=model, use_NN=True, get_T=True)

# For a symmetric T, Txy == Tyx. We'll just pick Txy = T[:, 0, 1].
Txx = new_T[:, 0, 0]
Tyy = new_T[:, 1, 1]
Txy = new_T[:, 0, 1]

# 1) Scatter plot of T_xx vs T_yy
plt.figure()
plt.scatter(Txx.cpu().numpy(), Tyy.cpu().numpy(), s=5, alpha=0.5)
plt.xlabel('T_xx')
plt.ylabel('T_yy')
plt.title('Scatter of T_xx vs T_yy')
plt.grid(True)
plt.show()

# 2) Histogram of T_xy
plt.figure()
plt.hist(Txy.cpu().numpy(), bins=50)
plt.xlabel('T_xy')
plt.ylabel('Count')
plt.title('Histogram of T_xy')
plt.grid(True)
plt.show()

torch.save(new_T, 'fourier_t.pt')
# torch.save(T1.detach(), 'fourier_t1.pt')
# torch.save(T2.detach(), 'fourier_t2.pt')
for _, angle in tqdm(enumerate(theta)):
    theta_i = torch.tensor(angle * torch.pi / 180.0,dtype=torch.float32)  # convert to radian


    R = rotation_matrix(theta_i).to(device)  # get rotation matrix
    R_T = R.t()  # get rotation matrix transpose
    # R = torch.eye(d, device=device)
    # R_T = R.t()


    alpha_total = torch.zeros((Num, num, s, s), dtype=torch.float32)

    for j, length_scale_x in enumerate(length_x):
        for k, length_scale_y in enumerate(length_y):
            for i in range(num):
                n = j * length_y.size + k

                # get average 2pcf and volume fraction
                phi_sum = 0
                S2_sum = None
                for l in range(realization_per_field):
                    microstructure_i = get_microstructure(mode='generate', size=s, mean=0.5,
                                                          length_scale_x=length_scale_x,
                                                          length_scale_y=length_scale_y,
                                                          seed=seed + i)
                    microstructure_i = torch.tensor(microstructure_i, dtype=torch.float32,device=device)
                    phi_i = torch.mean(microstructure_i)
                    S2_i = twopcf(microstructure_i)
                    phi_sum += phi_i
                    if S2_sum is None:
                        S2_sum = S2_i
                    else:
                        S2_sum += S2_i
                phi = phi_sum / realization_per_field
                S2 = S2_sum / realization_per_field
                beta = sce_pipeline.compute_beta()
                A2 = sce_pipeline.compute_A2(beta,phi,S2)
                D = sce_pipeline.compute_D(phi,A2)
                sigma = sce_pipeline.compute_Sigma(D=D,phi=phi)

                _, T_matrix_, indices = sce_pipeline.compute_A2(beta,phi,S2,get_T_and_A=True)


                # Repeat similar steps for T11 and T01

                T_matrix_ = new_T

                # Compute the midpoint index
                midpoint = s**2 // 2 + s //2 - 1

                # Compute the mean of the neighboring elements at the midpoint
                mean_element = (T_matrix_[midpoint - 1] + T_matrix_[midpoint]) / 2 + 10

                # Split the tensor into two parts: before and after the midpoint
                tensor_before = T_matrix_[:midpoint]
                tensor_after = T_matrix_[midpoint:]

                # Inject the mean element at the midpoint
                T_matrix = torch.cat((tensor_before, mean_element.unsqueeze(0), tensor_after), dim=0)
                T_matrix = T_matrix.reshape(s,s,2,2).permute(1,0,2,3).reshape(s**2,2,2)

                beta = (sigma1 - sigma0) / (sigma1 + (d - 1) * sigma0)
                omega = 2*torch.pi

                M,N,M_inv = get_AB(D,beta,phi,sigma0,device)
                M_inv_t_psi = torch.matmul(M_inv, T_matrix)
                # Compute M⁻¹ N
                M_inv_N = torch.matmul(M_inv, N)
                # Compute the first term
                first_term = (d / (omega * phi ** 2)) * torch.matmul(M_inv_t_psi, M_inv_N)

                # Compute the second term
                second_term = (d * sigma0 / (omega * phi ** 2)) * M_inv_t_psi

                # Combine the terms
                result = torch.matmul(R, (first_term - second_term))
                result = torch.matmul(result, R_T)[:,0,0]


                alpha = result.reshape(s, s).detach().cpu()
                alpha_total[n,i, :,:] = alpha.permute(1,0)
            if i == 0 and n==0 and angle==0:
                y = indices[:, 1]
                x = indices[:, 0]
                norm = torch.norm(indices * (1024 / 64), dim=1, p=2).detach().cpu()
                # Assuming T_matrix is of shape [N, 2, 2], theta is a 1D tensor of shape [N]
                theta = torch.atan2(y, x).detach().cpu()


                theta_ = torch.linspace(-np.pi, np.pi, 100)
                two_theta = 2 * theta_
                # Generate theta values
                cos2theta = torch.cos(two_theta)
                sin2theta = torch.sin(two_theta)




                t00 = T_matrix_[:, 0, 0].cpu().detach() * norm**2
                t11 = T_matrix_[:, 1, 1].cpu().detach() * norm**2
                t01 = T_matrix_[:, 0, 1].cpu().detach() * norm**2


                plt.figure(figsize=(8, 5))
                plt.scatter(theta.numpy(), t00.numpy(), color='r', alpha=0.6, label=r'$T_{00}$ data')
                plt.scatter(theta_.numpy(), cos2theta.numpy(), color='g', alpha=0.6, label=r'cos2theta')
                plt.xlabel(r'$\theta$ (radians)')
                plt.ylabel(r'$T_{00}$')
                plt.title(r'Scatter Plot of $T_{00}$ vs $\theta$')
                plt.legend()
                plt.grid(True)
                plt.show()


                plt.figure(figsize=(8, 5))
                plt.scatter(theta.numpy(), t01.numpy(), color='r', alpha=0.6, label=r'$T_{01}$ data')
                plt.scatter(theta_.numpy(), sin2theta.numpy(), color='g', alpha=0.6, label=r'sin2theta')
                plt.xlabel(r'$\theta$ (radians)')
                plt.ylabel(r'$T_{00}$')
                plt.title(r'Scatter Plot of $T_{01}$ vs $\theta$')
                plt.legend()
                plt.grid(True)
                plt.show()



                plt.figure(figsize=(8, 5))
                plt.scatter(theta.numpy(), t11.numpy(), color='r', alpha=0.6, label=r'$T_{11}$ data')
                plt.scatter(theta_.numpy(), -cos2theta.numpy(), color='g', alpha=0.6, label=r'-cos2theta')
                plt.xlabel(r'$\theta$ (radians)')
                plt.ylabel(r'$T_{00}$')
                plt.title(r'Scatter Plot of $T_{11}$ vs $\theta$')
                plt.legend()
                plt.grid(True)
                plt.show()




        mean_alpha = alpha_total.mean(dim=(0, 1))
        # mean_alpha = normalize(mean_alpha)
        np.save(f'./result_1617/optimal_w_{s}_{angle}.npy', mean_alpha)
        plt.figure(figsize=(8, 8))  # Set the figure size
        plt.imshow(mean_alpha.detach().cpu().numpy(), cmap='plasma', interpolation='nearest',
              vmin=np.percentile(mean_alpha, 0.5), vmax=np.percentile(mean_alpha, 99.5))
        # plt.gca().invert_yaxis()
        plt.colorbar(label='importance')
        plt.savefig(f'./result_1617/optimal_w_{s}_{angle}.png', dpi=300)