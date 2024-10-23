import numpy as np
import matplotlib.pyplot as plt
from src import SI_generated, window, Aepsilon, error
import sys

def main():

    # 设置测试数据参数
    dx = 0.05063
    dy = 0.05063
    Ne = 105
    radius = 1.0
    center = (2.5, 2.5)
    grid_size = 5.0

    # 测试
    eulerian_points, lagrangian_points, nearest_grid_points, delta_I, eta_I, all_S_I = SI_generated.generate_grid(
        dx, dy, Ne, radius, center, grid_size
    )

    all_modified_w = window.compute_all_modified_window_functions(
            all_S_I, lagrangian_points, nearest_grid_points, delta_I, eta_I, dx, dy
    )

    #sys.exit(0)

    delta_s_I = Aepsilon.compute_delta_s_I(radius, Ne)
    print(delta_s_I)

    A_matrix = Aepsilon.compute_A_matrix(
            delta_s_I, all_modified_w, dx, dy, lagrangian_points, delta_I, eta_I, all_S_I
    )

    #sys.exit(0)

    epsilon = Aepsilon.compute_epsilon(A_matrix)
    print(epsilon)
    print(f"... 共 {Ne} 个 ε 值")

    #sys.exit(0)    
    
    infinity_norm_error = error.compute_error(lagrangian_points, nearest_grid_points, all_modified_w, epsilon, delta_s_I)
    print(f"Error: {infinity_norm_error}")

if __name__ == "__main__":
    main()
