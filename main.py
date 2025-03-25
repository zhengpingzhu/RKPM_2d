import numpy as np
import matplotlib.pyplot as plt
from src import SI_generated, window, Aepsilon, error

def main():

    # 设置测试数据参数
    dx = 0.05063450674257659
    dy = 0.05063450674257659
    Ne = 105
    radius = 1.0
    center = (2.5, 2.5)
    grid_size = 5.0
    Delta_A = dx * dy
    max_iterations  = 100
    tolerance = 1e-6
    # 测试
    eulerian_points, lagrangian_points, nearest_grid_points, delta_I, eta_I, all_S_I = SI_generated.generate_grid(
        dx, dy, Ne, radius, center, grid_size
    )

    # print(delta_I[0],eta_I[0],dx,dy)

    all_modified_w = window.compute_all_modified_window_functions(
            all_S_I, lagrangian_points, dx, dy
    )

    delta_s_I = Aepsilon.compute_delta_s_I(radius, Ne)

    # A_matrix = Aepsilon.compute_A_matrix(
    #         delta_s_I, all_modified_w, dx, dy, lagrangian_points, all_S_I)
    # eigenvalues = np.linalg.eigvals(A_matrix)
    #print("A Matrix:")
    #print(np.min(eigenvalues),np.max(eigenvalues))

    epsilon = Aepsilon.solve_epsilon(Ne, max_iterations, tolerance, eulerian_points, all_S_I, all_modified_w, delta_s_I, Delta_A)
    print(f"ε的平均值:",np.mean(epsilon))
    
    infinity_norm_error = error.compute_error(eulerian_points, lagrangian_points, all_S_I, all_modified_w, epsilon, delta_s_I, Delta_A)
    print(f"Error: {infinity_norm_error}")

if __name__ == "__main__":
    main()
