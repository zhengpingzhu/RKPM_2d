import numpy as np
from scipy.spatial import KDTree

# 计算 si all
def generate_grid(dx, dy, Ne, radius, center, grid_size):

    # 生成欧拉网格
    x = np.arange(0, grid_size + dx, dx)
    y = np.arange(0, grid_size + dy, dy)
    X, Y_grid = np.meshgrid(x, y)

    # 获取欧拉坐标
    eulerian_points = np.vstack([X.ravel(), Y_grid.ravel()]).T

    # 生成拉格朗日点
    theta = np.linspace(0, 2 * np.pi, Ne, endpoint=False)
    lagrangian_points = np.vstack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ]).T

    # 构建 KDTree
    tree = KDTree(eulerian_points)

    # 查询每个拉格朗日点的最近欧拉网格点
    distances, nearest_indices = tree.query(lagrangian_points)
    nearest_grid_points = eulerian_points[nearest_indices]

    # 获取 X 和 Y 网格的形状
    grid_shape = X.shape

    # 初始化 delta_I 和 eta_I 数组
    delta_I = np.zeros(Ne)
    eta_I = np.zeros(Ne)

    # 遍历每个最近的欧拉网格点，计算 h_x_plus, h_x_minus, h_y_plus, h_y_minus
    for idx, nearest_idx in enumerate(nearest_indices):
        
        # 获取 i, j 索引
        i, j = np.unravel_index(nearest_idx, grid_shape)

        # 计算 h_x_plus 和 h_x_minus
        x_distance = np.full(6, -1.0)
        n = 0
        for k in [-1, 0, 1]:
            for l in [0, 1]:
                ni = i + k
                nj = j + l
                if (0 <= ni < X.shape[0]) and (0 <= nj < X.shape[1]) and (0 <= nj - 1 < X.shape[1]):
                    distance = X[ni, nj] - X[ni, nj - 1]
                    if distance >= 0:
                        x_distance[n] = distance
                        n += 1
        values_x = x_distance[x_distance >= 0]
        h_x_plus = np.max(values_x)
        h_x_minus = np.min(values_x)

        # 计算 h_y_plus 和 h_y_minus
        y_distance = np.full(6, -1.0)
        n = 0
        for k in [0, 1]:
            for l in [-1, 0, 1]:
                ni = i + k
                nj = j + l
                if (0 <= ni < Y_grid.shape[0]) and (0 <= ni - 1 < Y_grid.shape[0]) and (0 <= nj < Y_grid.shape[1]):
                    distance = Y_grid[ni, nj] - Y_grid[ni - 1, nj]
                    if distance >= 0:
                        y_distance[n] = distance
                        n += 1
        values_y = y_distance[y_distance >= 0]
        h_y_plus = np.max(values_y)
        h_y_minus = np.min(values_y)

        # 计算 delta_I 和 eta_I
        delta_I[idx] = (5 / 6) * h_x_plus + (1 / 6) * h_x_minus + (1 / 10) * dx
        eta_I[idx] = (5 / 6) * h_y_plus + (1 / 6) * h_y_minus + (1 / 10) * dy

    # 计算 all_S_I
    all_S_I = []
    for idx in range(Ne):

        nearest_point = nearest_grid_points[idx]
        delta_I_lag = delta_I[idx]
        eta_I_lag = eta_I[idx]

        # 找到位于矩形区域内的欧拉网格点
        mask_x = np.abs(eulerian_points[:, 0] - nearest_point[0]) < 1.5 * delta_I_lag
        mask_y = np.abs(eulerian_points[:, 1] - nearest_point[1]) < 1.5 * eta_I_lag

        S_I = eulerian_points[mask_x & mask_y]

        all_S_I.append(S_I)

    return eulerian_points, lagrangian_points, nearest_grid_points, delta_I, eta_I, all_S_I 
