import numpy as np
import math
from src import window
from scipy.sparse.linalg import bicgstab, LinearOperator

# 计算弧长
def compute_delta_s_I(radius, Ne):

    central_angle = 2 * math.pi / Ne
    delta_s_I = radius * central_angle
    return delta_s_I

# 计算矩阵A
def compute_A_matrix(delta_s_I, all_modified_w, dx, dy, lagrangian_points, all_S_I):

    Delta_A = dx * dy
    Ne = len(all_modified_w)
    
    # 初始化 A 矩阵
    A = np.zeros((Ne, Ne))
    
    for I in range(Ne):
        S_I = all_S_I[I]

        for K in range(Ne):
            # 计算 a_IK = delta_s_I * sum(w_I * w_K) * Delta_A
            nearest_grid_point = lagrangian_points[K]
            M_I = window.compute_m_ab_matrix(S_I, nearest_grid_point, dx, dy)
            d_I = window.compute_b_I(M_I)
            w_k = window.modified_window_function(S_I, nearest_grid_point, d_I, dx, dy)
            sum_w = np.sum(np.array(all_modified_w[I] * np.array(w_k)))
            A[I, K] = delta_s_I * sum_w * Delta_A
    
    return A

def implicit_A(eulerian_points, all_S_I, all_modified_w, epsilon, delta_s_I, delta_A):

    dispersion = np.zeros(len(eulerian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)      
        for j, point in enumerate(S_I):
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            dispersion[ide] += epsilon[idx] * modified_w[j]  * delta_s_I

    interpolate = np.zeros(len(epsilon))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, point in enumerate(S_I):
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            interpolate[idx] += dispersion[ide] * modified_w[j] * delta_A

    return interpolate

def solve_epsilon(Ne, max_iterations, tolerance, eulerian_points, all_S_I, all_modified_w, delta_s_I, delta_A):
    b = np.ones(Ne)
    iteration_count = 0

    def callback(xk):
        nonlocal iteration_count
        iteration_count += 1
        residual_norm = np.linalg.norm(b - A_operator(xk))
        print(f"Residual norm: {residual_norm:.6e}")

    def A_operator(epsilon):
        return implicit_A(eulerian_points, all_S_I, all_modified_w, epsilon, delta_s_I, delta_A)
    
    A_linop = LinearOperator((Ne, Ne), matvec=A_operator)
    
    epsilon, info = bicgstab(
        A_linop, b,
        rtol=tolerance,
        atol=1e-8,
        maxiter=max_iterations,
        x0=np.zeros(Ne),
        callback = callback
    )

    if info == 0:
        print("Converged in {} iterations".format(iteration_count))
    else:
        print("Did not converge within {} iterations".format(max_iterations))

    return epsilon

