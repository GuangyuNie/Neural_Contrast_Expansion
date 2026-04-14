# from scipy.sparse import csr_matrix, lil_matrix, diags
# import numpy as np
# class effective_conductivity_pde:
#
#     def __init__(self, microstructure, sigma0, sigma1):
#         # self.mat = mat
#         # self.size = size
#         # self.id = id
#         self.microstructure = microstructure
#         self.size = microstructure.shape[0]
#         self.sigma0 = sigma0
#         self.sigma1 = sigma1
#         # self.vf = vf
#
#     def setup_system(self, sigma):
#         n = self.size**2
#         A = lil_matrix((n, n))
#         b = np.zeros(n)
#
#         # Index helper to convert 2D to 1D index
#         def idx(i, j):
#             return i * self.size + j
#
#         # Setup the potential gradient to simulate a uniform electric field
#         E0 = 1  # Electric field magnitude in the x-direction
#
#         # Fill boundary conditions
#         for j in range(self.size):
#             # Top boundary (constant potential)
#             A[idx(0, j), idx(0, j)] = 1
#             b[idx(0, j)] = 1  # Set to 0 (or any other reference potential)
#
#             # Bottom boundary (constant potential)
#             A[idx(self.size - 1, j), idx(self.size - 1, j)] = 1
#             b[idx(self.size - 1, j)] = 0
#
#         for i in range(self.size):
#             # Left boundary (gradient)
#             A[idx(i, 0), idx(i, 0)] = 1
#             b[idx(i, 0)] = 0
#
#             # Right boundary (gradient)
#             A[idx(i, self.size - 1), idx(i, self.size - 1)] = 1
#             b[idx(i, self.size - 1)] = 0
#
#         # Fill the matrix A for interior points using vectorized operations
#         for i in range(1, self.size - 1):
#             for j in range(1, self.size - 1):
#                 index = idx(i, j)
#                 A[index, idx(i-1, j)] = (sigma[i-1, j] + sigma[i, j])/2
#                 A[index, idx(i+1, j)] = (sigma[i+1, j] + sigma[i, j])/2
#                 A[index, idx(i, j-1)] = (sigma[i, j-1] + sigma[i, j])/2
#                 A[index, idx(i, j+1)] = (sigma[i, j+1] + sigma[i, j])/2
#                 A[index, index] = - (2 * sigma[i, j] + (sigma[i-1, j] + sigma[i+1, j] + sigma[i, j-1] + sigma[i, j+1])/2)
#
#         A = A.tocsr()
#         return A, b
#
#     def solve_laplace_equation(self, A, b):
#         from scipy.sparse.linalg import spsolve
#         phi = spsolve(A, b)
#         # residual = np.linalg.norm(A.dot(phi) - b) / self.size**2
#         # print(f"Residual: {residual}")
#         return phi.reshape((self.size, self.size))
#
#     def setup_conductivity_matrix(self, sigma0, sigma1):
#         return np.where(self.microstructure == 0, sigma0, sigma1)
#
#     def compute_ensemble_flux(self, sigma, phi):
#
#         grad = np.gradient(phi)  # grad[0] (grad[1]) contains gradient in y (x) direction
#         # print(grad)
#         J = -np.array([sigma * grad[0], sigma * grad[1]])  # J_x and J_y components
#         # print(sigma)
#         # J_avg = np.mean(J[:,int(self.size/3):int(self.size*2/3),int(self.size/3):int(self.size*2/3)], axis=(1, 2))
#         J_avg = np.mean(J, axis=(1, 2))
#         # print(J.shape)
#         # print(J_avg.shape)
#         return J_avg
#
#     def compute_flux(self, sigma, phi):
#
#         grad = np.gradient(phi)  # grad[0] (grad[1]) contains gradient in y (x) direction
#         # print(grad)
#         J = -np.array([sigma * grad[0], sigma * grad[1]])  # J_x and J_y components
#         # print(sigma)
#         # J_avg = np.mean(J[:,int(self.size/3):int(self.size*2/3),int(self.size/3):int(self.size*2/3)], axis=(1, 2))
#         # print(J.shape)
#         # print(J_avg.shape)
#         return sigma * grad[0], sigma * grad[1]
#
#     def compute_electric_field(self, phi):
#         phi_mean = np.mean(phi, axis=1)
#         # print(phi_mean)
#         # print((phi_mean[0] - phi_mean[-1])/(self.size - 1))
#         return (phi_mean[0] - phi_mean[-1])/(self.size - 1)
#
#     def compute(self):
#         # Example usage
#         # microstructure = self.generate_microstructure()
#         sigma = self.setup_conductivity_matrix(self.sigma0, self.sigma1)
#
#         # Calculate for E-field along x-axis
#         A_x, b_x = self.setup_system(sigma.T)
#         phi_x = self.solve_laplace_equation(A_x, b_x)
#
#         # Calculate for E-field along y-axis
#         A_y, b_y = self.setup_system(sigma)
#         phi_y = self.solve_laplace_equation(A_y, b_y)
#
#         J_x = self.compute_ensemble_flux(sigma.T, phi_x)  # Sigma(1,1) Sigma (2,1)
#         J_y = self.compute_ensemble_flux(sigma, phi_y)  # Sigma(2,2) Sigma (1,2)
#
#         e0_x = self.compute_electric_field(phi_x)
#         e0_y = self.compute_electric_field(phi_y)
#         Sigma = np.array([[J_x[0]/e0_x, J_y[1]/e0_y], [J_x[1]/e0_x, J_y[0]/e0_y]])
#
#         return Sigma
#
#
#     def get_flux(self):
#         # Example usage
#         # microstructure = self.generate_microstructure()
#         sigma = self.setup_conductivity_matrix(self.sigma0, self.sigma1)
#
#         # Calculate for E-field along x-axis
#         A_x, b_x = self.setup_system(sigma.T)
#         phi_x = self.solve_laplace_equation(A_x, b_x)
#
#         # Calculate for E-field along y-axis
#         A_y, b_y = self.setup_system(sigma)
#         phi_y = self.solve_laplace_equation(A_y, b_y)
#
#         J_x, _ = self.compute_flux(sigma.T, phi_x)  # Sigma(1,1) Sigma (2,1)
#         J_y, _ = self.compute_flux(sigma, phi_y)  # Sigma(2,2) Sigma (1,2)
#
#         return -J_x, -J_y, phi_x, phi_y
#


import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class effective_conductivity_pde:
    def __init__(self, microstructure, sigma0, sigma1):
        """
        microstructure: 0/1 二值数组
        sigma0, sigma1: 两相的标量电导率
        """
        self.ms = microstructure.astype(np.int8)
        self.n = microstructure.shape[0]
        assert self.ms.shape[0] == self.ms.shape[1], "需要正方网格"
        self.s0 = float(sigma0)
        self.s1 = float(sigma1)
        # 单位网格间距 h=1 => 物理长度 L = n-1
        self.L = self.n - 1

    def sigma_field(self):
        return np.where(self.ms == 0, self.s0, self.s1).astype(float)

    @staticmethod
    def _harmonic(a, b, eps=1e-30):
        return 2.0 * a * b / (a + b + eps)

    def _assemble_A_b(self, sigma, drive='x'):
        """
        组装 A φ = b
        drive: 'x' 或 'y'，决定线性 Dirichlet 的方向
        """
        n = self.n
        idx = lambda i, j: i * n + j
        A = lil_matrix((n * n, n * n))
        b = np.zeros(n * n)

        # 先准备边界 Dirichlet 值（线性势）
        phi_bc = np.zeros((n, n), dtype=float)
        if drive == 'x':
            # 左右 0/1，上下依 x 线性：phi(i,j) = j/L
            for i in range(n):
                for j in range(n):
                    phi_bc[i, j] = 1.0 - j / self.L
        else:  # drive == 'y'
            # 上下 1/0，左右依 y 线性：phi(i,j) = 1 - i/L
            for i in range(n):
                for j in range(n):
                    phi_bc[i, j] = 1.0 - i / self.L

        # Dirichlet 边界打点
        is_boundary = np.zeros((n, n), dtype=bool)
        is_boundary[0, :] = True
        is_boundary[-1, :] = True
        is_boundary[:, 0] = True
        is_boundary[:, -1] = True

        # 内部：五点格式（谐均值）
        for i in range(n):
            for j in range(n):
                p = idx(i, j)
                if is_boundary[i, j]:
                    A[p, p] = 1.0
                    b[p] = phi_bc[i, j]
                else:
                    # 邻接谐均值（注意 i 是 y，j 是 x）
                    sN = self._harmonic(sigma[i, j], sigma[i-1, j])
                    sS = self._harmonic(sigma[i, j], sigma[i+1, j])
                    sW = self._harmonic(sigma[i, j], sigma[i, j-1])
                    sE = self._harmonic(sigma[i, j], sigma[i, j+1])

                    A[p, idx(i-1, j)] = -sN
                    A[p, idx(i+1, j)] = -sS
                    A[p, idx(i, j-1)] = -sW
                    A[p, idx(i, j+1)] = -sE
                    A[p, p] = (sN + sS + sW + sE)
                    # 内部无源 => b[p]=0

        return A.tocsr(), b, phi_bc

    def _solve_phi(self, A, b):
        phi = spsolve(A, b)
        return phi.reshape(self.n, self.n)

    def _average_flux_faces(self, sigma, phi):
        """
        用“面”通量做平均：
          Jx(i, j+1/2) = - sigma_e * (phi[i, j+1] - phi[i, j]) / h, h=1
          Jy(i+1/2, j) = - sigma_s * (phi[i+1, j] - phi[i, j]) / h
        返回 (Jx_avg, Jy_avg)
        """
        n = self.n

        # 水平面（x 向通量），大小 (n, n-1)
        sig_e = self._harmonic(sigma[:, :-1], sigma[:, 1:])
        dphix = (phi[:, 1:] - phi[:, :-1])
        Jx_faces = -sig_e * dphix

        # 垂直面（y 向通量），大小 (n-1, n)
        sig_s = self._harmonic(sigma[:-1, :], sigma[1:, :])
        dphiy = (phi[1:, :] - phi[:-1, :])
        Jy_faces = -sig_s * dphiy

        Jx_avg = Jx_faces.mean()
        Jy_avg = Jy_faces.mean()
        return Jx_avg, Jy_avg

    def compute(self):
        """
        做两次加载：
          1) drive='x'：施加 Ex ≈ 1/L，得平均 J=(Jx_x, Jy_x)
          2) drive='y'：施加 Ey ≈ 1/L，得平均 J=(Jx_y, Jy_y)
        然后 Σ 的两列分别是 J/E。
        """
        sigma = self.sigma_field()

        # x 方向外场
        A_x, b_x, _ = self._assemble_A_b(sigma, drive='x')
        phi_x = self._solve_phi(A_x, b_x)
        Jx_x, Jy_x = self._average_flux_faces(sigma, phi_x)

        # y 方向外场
        A_y, b_y, _ = self._assemble_A_b(sigma, drive='y')
        phi_y = self._solve_phi(A_y, b_y)
        Jx_y, Jy_y = self._average_flux_faces(sigma, phi_y)

        # 外场大小（线性 Dirichlet：左右差 1，域长 L => E = 1/L）
        E0 = 1.0 / self.L

        # 以两次加载的响应组 Σ 的两列
        Sigma = np.array([
            [Jx_x / E0, Jx_y / E0],
            [Jy_x / E0, Jy_y / E0],
        ])
        return Sigma

    # 如果你还想拿到点上“梯度通量场”，也可以保留这个接口：
    def get_flux_fields(self):
        sigma = self.sigma_field()
        A_x, b_x, _ = self._assemble_A_b(sigma, drive='x')
        phi_x = self._solve_phi(A_x, b_x)
        A_y, b_y, _ = self._assemble_A_b(sigma, drive='y')
        phi_y = self._solve_phi(A_y, b_y)

        # 点值梯度（注意顺序：grad[0]=d/dy, grad[1]=d/dx）
        gy_x, gx_x = np.gradient(phi_x)
        gy_y, gx_y = np.gradient(phi_y)

        Jx_field_x = -sigma * gx_x
        Jy_field_x = -sigma * gy_x
        Jx_field_y = -sigma * gx_y
        Jy_field_y = -sigma * gy_y

        return (Jx_field_x, Jy_field_x, phi_x), (Jx_field_y, Jy_field_y, phi_y)
