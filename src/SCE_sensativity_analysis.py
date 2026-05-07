from helper.utils import *
from sce_pipeline import *
from tqdm import tqdm

# load configs
config = load_config('config.yaml')
sigma0 = config['sigma0']
sigma1 = config['sigma1']
size = config['size']
d = config['d']
device_id = config['device']
npcf = config['n']
gt_size = config['gt_size']
device = torch.device(f"cuda:{device_id}")
mean = config['mean']
seed = config['seed']

# length_x = np.array([0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7])
# length_y = np.array([0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7])
sigma0 = 1.0
sigma1 = 2.0
mean = 0.5
length_x = np.array([0.001])
length_y = np.array([0.001])
num = 1
s = 63
realization_per_field = 30
N = length_x.size*length_y.size

torch.autograd.set_detect_anomaly(True)
sce_pipeline = EffectiveConductivityNPCF(
    s, sigma0, sigma1, 2, d, device, gt_size)

theta = np.arange(0, 361, 3)  # get theta in degree
for _, angle in tqdm(enumerate(theta)):
    theta_i = torch.tensor(angle * torch.pi / 180.0,dtype=torch.float32)  # convert to radian
    R = rotation_matrix(theta_i).to(device)  # get rotation matrix
    R_T = R.T  # get rotation matrix transpose
    alpha_total = torch.zeros((N, num, s, s), dtype=torch.float32)

    for j, length_scale_x in enumerate(length_x):
        for k, length_scale_y in enumerate(length_y):
            for i in range(num):
                n = j * length_y.size + k

                # get average 2pcf and volume fraction
                phi_sum = 0
                S2_sum = None
                for l in range(realization_per_field):
                    microstructure_i = get_microstructure(mode='generate', size=s, mean=mean,
                                                          length_scale_x=length_scale_x,
                                                          length_scale_y=length_scale_y,
                                                          seed=seed + i)
                    microstructure_i = torch.tensor(microstructure_i, dtype=torch.float32)
                    phi_i = torch.mean(microstructure_i)
                    S2_i = twopcf(microstructure_i)
                    phi_sum += phi_i
                    if S2_sum is None:
                        S2_sum = S2_i
                    else:
                        S2_sum += S2_i
                phi = phi_sum / realization_per_field
                S2 = S2_sum / realization_per_field

                S2_with_grad = S2.clone().detach().requires_grad_(True)  # require gradient for 2pcf
                # todo check what's wrong
                beta = sce_pipeline.compute_beta()
                A2, T_matrices, valid_indices_ = sce_pipeline.compute_A2(beta, phi, S2_with_grad,get_T_and_A=True)
                D = sce_pipeline.compute_D(phi, A2)
                g00, = torch.autograd.grad(A2[0, 0], S2_with_grad, retain_graph=True)
                g11, = torch.autograd.grad(A2[1, 1], S2_with_grad, retain_graph=True)

                max_abs = (g11 + g00).abs().max().item()

                S2_with_grad.grad = None
                A2[0][0].backward(retain_graph=True)
                alpha00 = S2_with_grad.grad
                S2_with_grad.grad = None
                S2_with_grad.grad = None
                A2[0][1].backward(retain_graph=True)
                alpha01 = S2_with_grad.grad
                S2_with_grad.grad = None
                A2[1][0].backward(retain_graph=True)
                alpha10 = S2_with_grad.grad
                S2_with_grad.grad = None
                A2[1][1].backward(retain_graph=True)
                alpha11 = S2_with_grad.grad
                alpha = alpha11.reshape(63, 63)
                ii = j = 63 // 2  # 31
                mid = alpha[ii, j]  
                alpha_3968 = np.delete(alpha11.ravel(), ii * 63 + j)
                test = (T_matrices * d / (2 * torch.pi))[:, 1, 1].detach().cpu() - alpha_3968
                A_INV = torch.inverse(D / beta ** 2 / (phi) ** 2 - torch.eye(d, device=device))
                device = A_INV.device
                dtype = A_INV.dtype
                output = sce_pipeline.compute_Sigma(D, phi)
                # row1 = torch.stack([alpha00.reshape(-1), alpha01.reshape(-1)], dim=1)  # (N, 2)
                # row2 = torch.stack([alpha10.reshape(-1), alpha11.reshape(-1)], dim=1)  # (N, 2)
                # alpha_stack = torch.stack([row1, row2], dim=1)  # (N, 2, 2)
                #
                # I2 = torch.eye(2, device=device, dtype=dtype)
                # M = (output.to(device=device, dtype=dtype) - sigma0 * I2)  # (2, 2)
                # alpha_stack = alpha_stack.to(device=device, dtype=dtype)
                # tmp = torch.matmul(A_INV, alpha_stack)  # (N, 2, 2)
                # GRAD = torch.matmul(tmp, M) * d / (2 * torch.pi) / phi**2  # (N, 2, 2)

                # output =  sce_pipeline.compute_based_on_S2(S2_with_grad, phi)
                # output_angled = torch.mm(R, torch.mm(output, R_T))[0][0]  # get Sigma_theta
                output_angled = (R @ output @ R_T)[0][0]
                S2_with_grad.grad = None
                output_angled.backward()
                alpha = S2_with_grad.grad
                alpha = alpha.reshape(s, s)
                # analytical_grad = (R @ GRAD @ R_T)[:,0,0]
                N = 63
                ii = j = N // 2
                k = ii * N + j

                mid_t = torch.tensor(0, device=device, dtype=dtype)
                # constant = analytical_grad.view(N, N).detach().reshape(N, N).cpu() / alpha.cpu()
                # alpha_flat = torch.cat([analytical_grad[:k], mid_t.view(1), analytical_grad[k:]], jdim=0)
                # alpha_63x63_t = alpha_flat.view(N, N)
                # alpha_total[n,i, :,:] = analytical_grad.view(N, N).detach()
                alpha_total[n,i, :,:] = alpha
                S2_with_grad.grad.zero_()

        mean_alpha = alpha_total.mean(dim=(0,1))
        np.save(f'./result_1617/optimal_w_{s}_{angle}.npy', mean_alpha)
        plt.figure(figsize=(8, 8))  # Set the figure size
        plt.gca().invert_yaxis()
        plt.imshow(mean_alpha.detach().cpu().numpy(), cmap='plasma', interpolation='nearest',
              vmin=np.percentile(mean_alpha, 0.5), vmax=np.percentile(mean_alpha, 99.5))
        plt.colorbar(label='importance')
        plt.savefig(f'./result_1617/optimal_w_{s}_{angle}.png', dpi=300)
