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
    x_i, y_j = nearest_grid_point
    delta_A_mn = dx * dy
    m_ab_matrix = np.zeros((6, 6))

    for x_mn, y_mn in S_I:
        delta_x = (x_mn - x_i) / dx
        delta_y = (y_mn - y_j) / dy
        dis_x = x_mn - x_i
        dis_y = y_mn - y_j

        w_total = window_function_d(delta_x) * window_function_d(delta_y)

        # 计算矩阵的每个元素
        m_ab_matrix[0, 0] += (dis_x**0.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[0, 1] += (dis_x**1.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[0, 2] += (dis_x**0.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[0, 3] += (dis_x**1.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[0, 4] += (dis_x**2.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[0, 5] += (dis_x**0.0) * (dis_y**2.0) * w_total * delta_A_mn

        m_ab_matrix[1, 0] += (dis_x**1.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[1, 1] += (dis_x**2.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[1, 2] += (dis_x**1.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[1, 3] += (dis_x**2.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[1, 4] += (dis_x**3.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[1, 5] += (dis_x**1.0) * (dis_y**2.0) * w_total * delta_A_mn

        m_ab_matrix[2, 0] += (dis_x**0.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[2, 1] += (dis_x**1.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[2, 2] += (dis_x**0.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[2, 3] += (dis_x**1.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[2, 4] += (dis_x**2.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[2, 5] += (dis_x**0.0) * (dis_y**3.0) * w_total * delta_A_mn

        m_ab_matrix[3, 0] += (dis_x**1.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[3, 1] += (dis_x**2.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[3, 2] += (dis_x**1.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[3, 3] += (dis_x**2.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[3, 4] += (dis_x**3.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[3, 5] += (dis_x**1.0) * (dis_y**3.0) * w_total * delta_A_mn

        m_ab_matrix[4, 0] += (dis_x**2.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[4, 1] += (dis_x**3.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[4, 2] += (dis_x**2.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[4, 3] += (dis_x**3.0) * (dis_y**1.0) * w_total * delta_A_mn
        m_ab_matrix[4, 4] += (dis_x**4.0) * (dis_y**0.0) * w_total * delta_A_mn
        m_ab_matrix[4, 5] += (dis_x**2.0) * (dis_y**2.0) * w_total * delta_A_mn

        m_ab_matrix[5, 0] += (dis_x**0.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[5, 1] += (dis_x**1.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[5, 2] += (dis_x**0.0) * (dis_y**3.0) * w_total * delta_A_mn
        m_ab_matrix[5, 3] += (dis_x**1.0) * (dis_y**3.0) * w_total * delta_A_mn
        m_ab_matrix[5, 4] += (dis_x**2.0) * (dis_y**2.0) * w_total * delta_A_mn
        m_ab_matrix[5, 5] += (dis_x**0.0) * (dis_y**4.0) * w_total * delta_A_mn

    return m_ab_matrix

# 计算bI 和 dI
def compute_b_I(M_I):

    e_1 = np.zeros(6)
    e_1[0] = 1

    d_I = np.linalg.solve(M_I, e_1)
    # try:
    #     d_I = np.linalg.solve(M_I, e_1)
    # except np.linalg.LinAlgError:
    #     d_I = np.linalg.lstsq(M_I, e_1, rcond=None)[0]

    return d_I

# 计算修正窗函数
def modified_window_function(S_I, nearest_grid_point, d_I, dx, dy):

    x_i, y_j = nearest_grid_point

    modified_w_values = []

    for x_mn, y_mn in S_I:
        delta_x = (x_mn - x_i) / dx
        delta_y = (y_mn - y_j) / dy
        dis_x = x_mn - x_i
        dis_y = y_mn - y_j

        # 计算窗函数
        w_total = window_function_d(delta_x) * window_function_d(delta_y)

        # 计算修正窗函数
        modified_w = d_I[0] * w_total + \
                     d_I[1] * dis_x * w_total + \
                     d_I[2] * dis_y * w_total + \
                     d_I[3] * dis_x * dis_y * w_total + \
                     d_I[4] * dis_x**2 * w_total + \
                     d_I[5] * dis_y**2 * w_total

        modified_w_values.append(modified_w)

    # integral = np.sum(np.array(modified_w_values) * dx * dy)
    # print(f"integral",integral)    

    return modified_w_values

# 对所有的点计算修正窗函数
def compute_all_modified_window_functions(all_S_I, lagrangian_points, dx, dy):

    all_modified_w = []
    for idx in range(len(lagrangian_points)):
        S_I = all_S_I[idx]
        nearest_grid_point = lagrangian_points[idx]

        # 计算 m_ab_matrix
        M_I = compute_m_ab_matrix(S_I, nearest_grid_point, dx, dy)

        # 计算 b_I 和 d_I
        d_I = compute_b_I(M_I)

        # 计算修正窗函数
        modified_w = modified_window_function(S_I, nearest_grid_point, d_I, dx, dy)

        all_modified_w.append(modified_w)

    return all_modified_w