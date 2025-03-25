import numpy as np

# 测试函数
def test_function(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

# 计算无穷范数误差
def compute_infinity_norm_error(original_values, interpolated_values):
    diff = original_values - interpolated_values
    return np.max(np.abs(diff))

# 计算误差
def compute_error(eulerian_points, lagrangian_points, all_S_I, all_modified_w, epsilon, delta_s_I, delta_A):
    original_values = test_function(lagrangian_points[:, 0], lagrangian_points[:, 1])

    dispersion_values = np.zeros(len(eulerian_points))
    for idx, (lagrangian_point, S_I, modified_w) in enumerate(zip(lagrangian_points, all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, point in enumerate(S_I):
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            dispersion_values[ide] += original_values[idx] * modified_w[j] * epsilon[idx] * delta_s_I

    interpolated_values = np.zeros(len(lagrangian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, point in enumerate(S_I):
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            interpolated_values[idx] += dispersion_values[ide] * modified_w[j] * delta_A

    return compute_infinity_norm_error(original_values, interpolated_values)