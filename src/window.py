import numpy as np

# 计算窗函数
def window_function_d(r):
    abs_r = np.abs(r)
    if 0.5 <= abs_r <= 1.5:
        return (1/6) * (5 - 3*abs_r - np.sqrt(-3*(1-abs_r)**2 + 1))
    elif abs_r <= 0.5:
        return (1/3) * (1 + np.sqrt(-3*r**2 + 1))
    else:
        return 0

# 计算矩阵m
def compute_m_ab_matrix(S_I, nearest_grid_point, dx, dy):

    m_ab_matrix = np.zeros((6,6))
    x_i, y_j = nearest_grid_point

    for point in S_I:
        x_mn, y_mn = point
        delta_x = (x_mn - x_i) / dx
        delta_y = (y_mn - y_j) / dy

        # 计算窗函数
        w_delta = window_function_d(delta_x)
        w_eta = window_function_d(delta_y)
        w_total = w_delta * w_eta
        
        # 面元
        delta_A_mn = dx * dy

        # 计算矩阵的每个元素
        m_ab_matrix[0, 0] += (delta_x**0.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[0, 1] += (delta_x**1.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[0, 2] += (delta_x**0.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[0, 3] += (delta_x**1.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[0, 4] += (delta_x**2.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[0, 5] += (delta_x**0.0) * (delta_y**2.0) * w_total * delta_A_mn

        m_ab_matrix[1, 0] += (delta_x**1.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[1, 1] += (delta_x**2.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[1, 2] += (delta_x**1.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[1, 3] += (delta_x**2.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[1, 4] += (delta_x**3.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[1, 5] += (delta_x**1.0) * (delta_y**2.0) * w_total * delta_A_mn

        m_ab_matrix[2, 0] += (delta_x**0.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[2, 1] += (delta_x**1.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[2, 2] += (delta_x**0.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[2, 3] += (delta_x**1.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[2, 4] += (delta_x**2.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[2, 5] += (delta_x**0.0) * (delta_y**3.0) * w_total * delta_A_mn

        m_ab_matrix[3, 0] += (delta_x**1.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[3, 1] += (delta_x**2.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[3, 2] += (delta_x**1.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[3, 3] += (delta_x**2.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[3, 4] += (delta_x**3.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[3, 5] += (delta_x**1.0) * (delta_y**3.0) * w_total * delta_A_mn

        m_ab_matrix[4, 0] += (delta_x**2.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[4, 1] += (delta_x**3.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[4, 2] += (delta_x**2.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[4, 3] += (delta_x**3.0) * (delta_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[4, 4] += (delta_x**4.0) * (delta_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[4, 5] += (delta_x**2.0) * (delta_y**2.0) * w_total * delta_A_mn

        m_ab_matrix[5, 0] += (delta_x**0.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[5, 1] += (delta_x**1.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[5, 2] += (delta_x**0.0) * (delta_y**3.0) * w_total * delta_A_mn
        m_ab_matrix[5, 3] += (delta_x**1.0) * (delta_y**3.0) * w_total * delta_A_mn
        m_ab_matrix[5, 4] += (delta_x**2.0) * (delta_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[5, 5] += (delta_x**0.0) * (delta_y**4.0) * w_total * delta_A_mn

    return m_ab_matrix

# 计算HI
def compute_H_I(delta_I, eta_I):

    H_I = np.diag([
        1,
        1 / delta_I,
        1 / eta_I,
        1 / (delta_I * eta_I),
        1 / (delta_I ** 2),
        1 / (eta_I ** 2)
    ])
    return H_I

# 计算bI 和 dI
def compute_b_I(H_I, M_I):

    # e_1 是 R^6 的标准基
    e_1 = np.zeros(6)
    e_1[0] = 1

    # 计算 H_I * M_I
    H_M = np.linalg.pinv(np.dot(H_I, M_I))
    # H_M = np.dot(H_I, M_I)

    # 计算 c_I
    c_I = np.dot(H_M, e_1)
    # c_I = np.linalg.solve(H_M, e_1)

    # 计算 b_I
    b_I = np.dot(H_I, c_I)

    # 计算 d_I
    M_pinv = np.linalg.pinv(M_I)
    d_I = np.dot(M_pinv, e_1)
    # d_I, residuals, rank, s = np.linalg.lstsq(M_I, e_1, rcond=None)
    # d_I = np.linalg.solve(M_I, e_1)

    #print('M_I')
    #print(M_I)
    #print('H_I')
    #print(H_I)

    return b_I, d_I

# 计算修正窗函数
def modified_window_function(S_I, nearest_grid_point, d_I, dx, dy):

    x_i, y_j = nearest_grid_point 

    modified_w_values = []

    for point in S_I:
        x_mn, y_mn = point  
        delta_x = (x_mn - x_i) / dx
        delta_y = (y_mn - y_j) / dy

        # 计算窗函数 
        w_delta = window_function_d(delta_x)
        w_eta = window_function_d(delta_y)
        w_total = w_delta * w_eta

        # 计算修正窗函数
        modified_w = d_I[0] * w_total #0
        modified_w = modified_w + d_I[1] * delta_x * w_total
        modified_w = modified_w + d_I[2] * delta_y * w_total
        modified_w = modified_w + d_I[3] * delta_x * delta_y * w_total
        modified_w = modified_w + d_I[4] * delta_x ** 2 * w_total
        modified_w = modified_w + d_I[5] * delta_y ** 2 * w_total

        modified_w_values.append(modified_w)
    
    return modified_w_values

# 对所有的点计算修正窗函数
def compute_all_modified_window_functions(all_S_I, lagrangian_points, nearest_euler_points, delta_I, eta_I, dx, dy):

    all_modified_w = []
    for idx in range(len(lagrangian_points)):
        S_I = all_S_I[idx]
        nearest_grid_point = lagrangian_points[idx] #nearest_euler_points[idx]
        delta_I_lag = delta_I[idx]
        eta_I_lag = eta_I[idx]

        # 计算 m_ab_matrix
        M_I = compute_m_ab_matrix(S_I, nearest_grid_point, dx, dy)
        # M_I = compute_m_ab_matrix(S_I, nearest_grid_point, lagrangian_points[idx], dx, dy)

        # 计算 H_I
        H_I = compute_H_I(delta_I_lag, eta_I_lag)

        # 计算 b_I 和 d_I
        b_I, d_I = compute_b_I(H_I, M_I)

        # 计算修正窗函数
        modified_w = modified_window_function(S_I, nearest_grid_point, d_I, dx, dy)
        # modified_w = modified_window_function(S_I, nearest_grid_point, b_I, dx, dy)

        all_modified_w.append(modified_w)
    
    return all_modified_w
