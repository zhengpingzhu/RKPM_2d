import numpy as np
import math

# 测试函数
def test_function(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

# 计算无穷范数误差
def compute_infinity_norm_error(original_values, interpolated_values):
    return np.linalg.norm(original_values - interpolated_values, ord=np.inf)

# 计算误差
def compute_error(lagrangian_points, nearest_grid_points, all_modified_w, epsilon, delta_s_I):

    # 计算测试函数在拉格朗日点的值
    original_values = test_function(lagrangian_points[:, 0], lagrangian_points[:, 1])

    # 计算插值后的值 g^IC
    interpolated_values = []
    for idx, nearest_point in enumerate(nearest_grid_points):

        # 将 modified_w 转换为 numpy 数组
        modified_w = np.array(all_modified_w[idx]) 

        # 计算插值
        interpolated_value = np.sum(modified_w * epsilon[idx] * delta_s_I)  
        interpolated_values.append(interpolated_value)
    
    interpolated_values = np.array(interpolated_values)

    # 计算误差
    infinity_norm_error = compute_infinity_norm_error(original_values, interpolated_values)
    
    return infinity_norm_error
