import numpy as np
import math
from src import window

# 计算弧长
def compute_delta_s_I(radius, Ne):

    central_angle = 2 * math.pi / Ne
    delta_s_I = radius * central_angle
    return delta_s_I

# 计算矩阵A
def compute_A_matrix(delta_s_I, all_modified_w, dx, dy, lagrangian_points, delta_I, eta_I, all_S_I):

    Delta_A = dx * dy
    Ne = len(all_modified_w)
    
    # 初始化 A 矩阵
    A = np.zeros((Ne, Ne))
    
    for I in range(Ne):
        S_I = all_S_I[I]
        # nearest_grid_point = lagrangian_points[I]
        delta_I_lag = delta_I[I]
        eta_I_lag = eta_I[I]
        for K in range(Ne):
            # 计算 a_IK = delta_s_I * sum(w_I * w_K) * Delta_A
            nearest_grid_point = lagrangian_points[K]
            M_I = window.compute_m_ab_matrix(S_I, nearest_grid_point, dx, dy)
            # print('Matrix A')
            # print(M_I)
            H_I = window.compute_H_I(delta_I_lag, eta_I_lag)
            b_I, d_I = window.compute_b_I(M_I, H_I)
            w_k = window.modified_window_function(S_I, nearest_grid_point, d_I, dx, dy)
            sum_w = np.sum(np.array(all_modified_w[I] * np.array(w_k)))
            A[I, K] = delta_s_I * sum_w * Delta_A
    
    return A

# 计算epsilon
def compute_epsilon(A):

    epsilon = np.linalg.solve(A, np.ones(A.shape[0]))
    #epsilon, residuals, rank, s = np.linalg.lstsq(A, np.ones(A.shape[0]), rcond=None)
    return epsilon

